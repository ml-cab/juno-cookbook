package cab.ml.juno.cookbook;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Scanner;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.node.LoraTrainableHandler;
import cab.ml.juno.player.ChatHistory;
import cab.ml.juno.player.ChatModelType;
import cab.ml.juno.player.LoraTrainer;
import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * In-process LoRA fine-tuning and inference session.
 *
 * <p>
 * Wraps {@link LoraTrainer} (training) and {@link GenerationLoop} (inference)
 * in a single {@link AutoCloseable} handle. The base model weights are never
 * modified; only the low-rank A/B adapter matrices are trained and persisted.
 *
 * <p>
 * Usage — train then chat:
 *
 * <pre>{@code
 * try (LoRATuner tuner = LoRATuner.builder(Path.of("/path/to/model.gguf")).build()) {
 *
 * 	// teach a fact
 * 	tuner.trainQA("What is your name?", "Juno");
 * 	tuner.save();
 *
 * 	// verify recall
 * 	String reply = tuner.chat("What is your name?");
 * 	System.out.println(reply);
 * }
 * }</pre>
 *
 * <p>
 * Usage — load saved adapter and run inference only:
 *
 * <pre>{@code
 * try (LoRATuner tuner = LoRATuner.builder(modelPath).adapterPath(Path.of("model.lora")).build()) {
 * 	String reply = tuner.chat("What is your name?");
 * }
 * }</pre>
 */
public final class LoRATuner implements AutoCloseable {

	/** Default LoRA rank. */
	static final int DEFAULT_RANK = 8;

	/** Default gradient steps per {@link #trainQA} call. */
	static final int DEFAULT_STEPS_QA = 10;

	/** Default gradient steps per {@link #trainText} call. */
	static final int DEFAULT_STEPS_TEXT = 50;

	/** Default early-stop loss threshold (0 = disabled). */
	static final float DEFAULT_EARLY_STOP = 0.25f;

	private final LoraTrainer trainer;
	private final GenerationLoop loop;
	private final String modelType;
	private final SamplingParams samplingParams;
	private final int stepsQa;
	private final int stepsText;
	private final float earlyStop;
	private final int chunkTokens;

	private ChatHistory history;

	private LoRATuner(LoraTrainer trainer, GenerationLoop loop, String modelType, SamplingParams samplingParams,
			int stepsQa, int stepsText, float earlyStop, int chunkTokens) {
		this.trainer = trainer;
		this.loop = loop;
		this.modelType = modelType;
		this.samplingParams = samplingParams;
		this.stepsQa = stepsQa;
		this.stepsText = stepsText;
		this.earlyStop = earlyStop;
		this.chunkTokens = chunkTokens;
		this.history = new ChatHistory();
	}

	// ── Training ──────────────────────────────────────────────────────────────

	/**
	 * Trains on a question/answer pair using the model's own chat template.
	 *
	 * <p>
	 * Four phrasings of the question are generated automatically to improve recall
	 * generalisation. The same chat template used at inference is applied during
	 * training so token distributions match.
	 *
	 * @param question user question (trailing {@code ?} added if absent)
	 * @param answer   expected assistant answer
	 * @return loss after the last gradient step, or {@link Float#NaN} if the input
	 *         was too short to train on
	 */
	public float trainQA(String question, String answer) {
		if (question == null || question.isBlank()) {
			throw new IllegalArgumentException("question must not be blank");
		}
		if (answer == null || answer.isBlank()) {
			throw new IllegalArgumentException("answer must not be blank");
		}
		return trainer.trainQaPair(question, answer, modelType, stepsQa);
	}

	/**
	 * Trains on raw text (no chat template applied).
	 *
	 * <p>
	 * Use {@link #trainQA} instead for factual Q&amp;A recall — the template
	 * context matters for instruction-tuned models.
	 *
	 * @param text training text; must tokenize to at least 2 tokens
	 * @return loss after the last gradient step, or {@link Float#NaN} if too short
	 */
	public float trainText(String text) {
		if (text == null || text.isBlank()) {
			throw new IllegalArgumentException("text must not be blank");
		}
		return trainer.trainRawText(text, stepsText, chunkTokens);
	}

	/**
	 * Loads training text from {@code file} and calls {@link #trainText}.
	 *
	 * @param file path to a text file
	 * @return loss after the last gradient step
	 * @throws IOException if the file cannot be read
	 */
	public float trainFile(Path file) throws IOException {
		if (file == null) {
			throw new IllegalArgumentException("file must not be null");
		}
		return trainText(Files.readString(file));
	}

	/**
	 * Saves adapter weights to the path configured at build time.
	 *
	 * @throws IOException if the write fails
	 */
	public void save() throws IOException {
		trainer.save();
	}

	/**
	 * Returns the path where {@link #save()} persists the adapter checkpoint.
	 */
	public Path adapterPath() {
		return trainer.adapterPath();
	}

	/**
	 * Returns the current {@link LoraAdapterSet} (live reference, not a copy).
	 */
	public LoraAdapterSet adapters() {
		return trainer.adapters();
	}

	// ── Inference ─────────────────────────────────────────────────────────────

	/**
	 * Sends one user turn and returns the assistant reply.
	 *
	 * <p>
	 * LoRA adapter weights are applied during inference via the underlying
	 * {@link LoraTrainableHandler}. KV cache is reused across turns.
	 *
	 * @param userText non-blank user message
	 * @return assistant reply text
	 * @throws IllegalArgumentException if userText is blank
	 */
	public String chat(String userText) {
		if (userText == null || userText.isBlank()) {
			throw new IllegalArgumentException("userText must not be blank");
		}
		history.addUser(userText);
		InferenceRequest request = InferenceRequest.ofSession(history.sessionId(), modelType, history.getMessages(),
				samplingParams, RequestPriority.NORMAL);
		GenerationResult result = loop.generate(request, TokenConsumer.discard());
		history.addAssistant(result.text());
		return result.text();
	}

	/**
	 * Resets conversation history and starts a fresh KV-cache session. The model
	 * and adapter weights remain intact; only the conversation context is cleared.
	 */
	public void resetHistory() {
		loop.evictSession(history.sessionId());
		history = new ChatHistory();
	}

	/**
	 * Returns the current session ID (stable until {@link #resetHistory()} is
	 * called).
	 */
	public String sessionId() {
		return history.sessionId();
	}

	/**
	 * Returns an unmodifiable snapshot of the current message history.
	 */
	public List<ChatMessage> history() {
		return history.getMessages();
	}

	/**
	 * Runs a blocking interactive REPL loop reading from {@code in} and writing to
	 * {@code out}. Type {@code exit} or {@code quit} to end. Returns normally on
	 * exit or EOF.
	 */
	public void runInteractive(InputStream in, PrintStream out) {
		try (Scanner scanner = new Scanner(in)) {
			out.println("LoRATuner ready. Type your message, or 'exit' to quit.");
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine().strip();
				if (line.isEmpty()) {
					continue;
				}
				if (line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit")) {
					break;
				}
				out.print("you> ");
				out.println(line);
				out.print("bot> ");
				out.flush();
				out.println(chat(line));
			}
		}
	}

	/**
	 * Releases the LoRA handler GPU resources and evicts the KV-cache session. The
	 * instance must not be used after this call.
	 */
	@Override
	public void close() {
		loop.evictSession(history.sessionId());
		trainer.close();
	}

	// ── Builder ───────────────────────────────────────────────────────────────

	public static Builder builder(Path modelPath) {
		return new Builder(modelPath);
	}

	public static final class Builder {

		private final Path modelPath;
		private Path adapterPath;
		private int rank = DEFAULT_RANK;
		private float alpha = -1f; // sentinel: default to rank
		private double lr = 1e-4;
		private int stepsQa = DEFAULT_STEPS_QA;
		private int stepsText = DEFAULT_STEPS_TEXT;
		private float earlyStop = DEFAULT_EARLY_STOP;
		private int chunkTokens = 32;
		private SamplingParams samplingParams = SamplingParams.defaults();

		private Builder(Path modelPath) {
			if (modelPath == null) {
				throw new IllegalArgumentException("modelPath must not be null");
			}
			this.modelPath = modelPath;
		}

		/**
		 * Path for loading/saving the .lora adapter checkpoint. Defaults to
		 * {@code <model-stem>.lora} in the same directory as the model.
		 */
		public Builder adapterPath(Path adapterPath) {
			this.adapterPath = adapterPath;
			return this;
		}

		/**
		 * Low-rank bottleneck dimension. Defaults to {@value #DEFAULT_RANK}. Common
		 * values: 4 (fast experiments), 8 (general use), 16 (complex domains).
		 */
		public Builder rank(int rank) {
			if (rank < 1) {
				throw new IllegalArgumentException("rank must be >= 1");
			}
			this.rank = rank;
			return this;
		}

		/**
		 * LoRA alpha scaling factor. Defaults to the same value as {@code rank}. The
		 * effective delta scale is {@code alpha / rank}.
		 */
		public Builder alpha(float alpha) {
			this.alpha = alpha;
			return this;
		}

		/**
		 * Adam learning rate. Defaults to {@code 1e-4}.
		 */
		public Builder lr(double lr) {
			if (lr <= 0) {
				throw new IllegalArgumentException("lr must be > 0");
			}
			this.lr = lr;
			return this;
		}

		/**
		 * Gradient steps per {@link #trainQA} call. Defaults to
		 * {@value #DEFAULT_STEPS_QA}.
		 */
		public Builder stepsQa(int stepsQa) {
			this.stepsQa = Math.max(1, stepsQa);
			return this;
		}

		/**
		 * Gradient steps per {@link #trainText} call. Defaults to
		 * {@value #DEFAULT_STEPS_TEXT}.
		 */
		public Builder stepsText(int stepsText) {
			this.stepsText = Math.max(1, stepsText);
			return this;
		}

		/**
		 * Stop training when loss drops below this threshold (prevents overfitting).
		 * Set to {@code 0} to disable. Defaults to {@value #DEFAULT_EARLY_STOP}.
		 */
		public Builder earlyStop(float earlyStop) {
			this.earlyStop = earlyStop;
			return this;
		}

		/**
		 * Max tokens per training chunk. Smaller values are faster per step on CPU.
		 * Defaults to 32.
		 */
		public Builder chunkTokens(int chunkTokens) {
			this.chunkTokens = Math.max(2, chunkTokens);
			return this;
		}

		/**
		 * Inference sampling parameters. Defaults to {@link SamplingParams#defaults()}.
		 */
		public Builder samplingParams(SamplingParams samplingParams) {
			if (samplingParams == null) {
				throw new IllegalArgumentException("samplingParams must not be null");
			}
			this.samplingParams = samplingParams;
			return this;
		}

		/**
		 * Loads the GGUF model, wires the LoRA training handler, and constructs the
		 * inference loop.
		 *
		 * <p>
		 * If {@code adapterPath} points to an existing file, the checkpoint is loaded.
		 * Otherwise new adapters are initialised from scratch (B = 0, A ~ N(0, 0.01)).
		 *
		 * @throws IOException if the model file cannot be read
		 */
		public LoRATuner build() throws IOException {
			System.setProperty("juno.byteOrder", "BE");

			float resolvedAlpha = alpha < 0 ? rank : alpha;
			Path resolvedAdapterPath = adapterPath != null ? adapterPath : defaultAdapterPath(modelPath);

			LoraTrainer trainer = LoraTrainer.open(modelPath, resolvedAdapterPath, rank, resolvedAlpha, lr);

			// Wire a separate LocalInferencePipeline backed by the same
			// LoraTrainableHandler
			// so that adapter weights applied during training are immediately visible at
			// inference.
			LlamaConfig config;
			Tokenizer tokenizer;
			try (GgufReader reader = GgufReader.open(modelPath)) {
				config = LlamaConfig.from(reader);
				tokenizer = GgufTokenizer.load(reader);
			}

			ShardAssignment assignment = new ShardAssignment("lora-node", "localhost", 0, 0, config.numLayers(), true,
					true);
			ShardMap shardMap = new ShardMap("model", config.numLayers(), List.of(assignment), Instant.now());

			LocalInferencePipeline pipeline = LocalInferencePipeline.from(shardMap, List.of(trainer.handler()),
					config.vocabSize(), config.hiddenDim(), config.numHeads());

			KVCacheManager kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));

			GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

			String modelType = ChatModelType.fromPath(modelPath.toString());

			return new LoRATuner(trainer, loop, modelType, samplingParams, stepsQa, stepsText, earlyStop, chunkTokens);
		}

		private static Path defaultAdapterPath(Path modelPath) {
			Path abs = modelPath.toAbsolutePath();
			String name = abs.getFileName().toString();
			int dot = name.lastIndexOf('.');
			String stem = dot > 0 ? name.substring(0, dot) : name;
			Path parent = abs.getParent();
			return parent != null ? parent.resolve(stem + ".lora") : Path.of(stem + ".lora");
		}
	}
}