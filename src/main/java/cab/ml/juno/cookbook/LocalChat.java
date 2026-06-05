package cab.ml.juno.cookbook;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
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
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.node.MatVec;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.player.ChatHistory;
import cab.ml.juno.player.ChatModelType;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * In-process jUno inference session.
 *
 * <p>
 * Loads a GGUF model into the local JVM (same wiring as {@code ./juno local}),
 * maintains a multi-turn chat history with KV-cache reuse across turns, and
 * exposes a single {@link #chat(String)} method for programmatic use. No
 * external server, no forked processes.
 *
 * <p>
 * Usage:
 *
 * <pre>{@code
 * try (LocalChat repl = LocalChat.builder(Path.of("/path/to/model.gguf")).build()) {
 *     String reply = repl.chat("What is 2 + 2?");
 *     System.out.println(reply);
 * }
 * }</pre>
 *
 * <p>
 * For an interactive terminal session backed by the same model:
 *
 * <pre>{@code
 * repl.runInteractive(System.in, System.out);
 * }</pre>
 *
 * <p>
 * To apply a trained LoRA adapter at inference time (equivalent to
 * {@code ./juno local --lora-play model.lora}):
 *
 * <pre>{@code
 * try (LocalChat repl = LocalChat.builder(Path.of("/path/to/model.gguf"))
 *         .loraPlay(Path.of("/path/to/model.lora"))
 *         .build()) {
 *     String reply = repl.chat("What is my name?");
 * }
 * }</pre>
 */
public final class LocalChat implements AutoCloseable {

	/**
	 * Default system prompt prepended to every request. Guides instruction-tuned
	 * models (e.g. TinyLlama-Chat) to maintain conversation context across turns.
	 */
	static final String DEFAULT_SYSTEM_PROMPT =
			"You are a helpful assistant. Remember everything the user tells you and refer back to it accurately.";

	private final GenerationLoop loop;
	private final String modelType;
	private final SamplingParams samplingParams;
	private final List<ForwardPassHandler> handlers;
	private final GpuContext gpuContext;
	private final String systemPrompt;

	private ChatHistory history;

	private LocalChat(GenerationLoop loop, String modelType, SamplingParams samplingParams,
			List<ForwardPassHandler> handlers, GpuContext gpuContext, String systemPrompt) {
		this.loop = loop;
		this.modelType = modelType;
		this.samplingParams = samplingParams;
		this.handlers = handlers;
		this.gpuContext = gpuContext;
		this.systemPrompt = systemPrompt;
		this.history = new ChatHistory();
	}

	/**
	 * Sends one user turn, accumulates reply in the session history, and returns
	 * the reply text.
	 *
	 * <p>
	 * KV cache is keyed on the stable session ID so prior context is reused without
	 * re-running prefill from scratch each turn.
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
		InferenceRequest request = InferenceRequest.ofSession(
				history.sessionId(), modelType, buildMessages(), samplingParams, RequestPriority.NORMAL);
		GenerationResult result = loop.generate(request, TokenConsumer.discard());
		history.addAssistant(result.text());
		return result.text();
	}

	/**
	 * Resets conversation history and starts a fresh KV-cache session. The model
	 * remains loaded; only the context is discarded.
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
	 * Runs a blocking interactive REPL loop, reading lines from {@code in} and
	 * writing replies to {@code out}. Type {@code exit} or {@code quit} to end the
	 * loop. This method returns normally when the user exits or {@code in} reaches
	 * EOF.
	 *
	 * <p>
	 * For a standard terminal:
	 *
	 * <pre>{@code
	 * repl.runInteractive(System.in, System.out);
	 * }</pre>
	 */
	public void runInteractive(InputStream in, PrintStream out) {
		try (Scanner scanner = new Scanner(in)) {
			out.println("jUno ready. Type your message, or 'exit' to quit.");
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
	 * Evicts the active KV-cache session and releases GPU resources. The instance
	 * must not be used after this call.
	 */
	@Override
	public void close() {
		loop.evictSession(history.sessionId());
		for (ForwardPassHandler handler : handlers) {
			handler.releaseGpuResources();
		}
		if (gpuContext != null) {
			gpuContext.close();
		}
	}

	/**
	 * Builds the message list for the current request. Prepends the system prompt
	 * when one is configured, so it is present on every turn without being stored
	 * in ChatHistory (which only tracks user/assistant turns).
	 */
	private List<ChatMessage> buildMessages() {
		List<ChatMessage> turns = history.getMessages();
		if (systemPrompt == null || systemPrompt.isBlank()) {
			return turns;
		}
		List<ChatMessage> messages = new ArrayList<>(turns.size() + 1);
		messages.add(ChatMessage.system(systemPrompt));
		messages.addAll(turns);
		return List.copyOf(messages);
	}

	// ── Builder ───────────────────────────────────────────────────────────────

	public static Builder builder(Path modelPath) {
		return new Builder(modelPath);
	}

	public static final class Builder {

		private final Path modelPath;
		private int nodeCount = 3;
		private boolean useGpu = true;
		private SamplingParams samplingParams = SamplingParams.defaults();
		private String systemPrompt = DEFAULT_SYSTEM_PROMPT;
		private Path loraPlayPath = null;

		private Builder(Path modelPath) {
			if (modelPath == null) {
				throw new IllegalArgumentException("modelPath must not be null");
			}
			this.modelPath = modelPath;
		}

		/**
		 * Number of in-process pipeline shards. Defaults to 3 (mirrors
		 * {@code ./juno local} defaults). Use 1 for the smallest memory footprint.
		 */
		public Builder nodeCount(int nodeCount) {
			this.nodeCount = Math.max(1, nodeCount);
			return this;
		}

		/**
		 * Whether to attempt GPU (CUDA) acceleration. Falls back to CPU automatically
		 * if CUDA is not available. Defaults to {@code true}.
		 */
		public Builder useGpu(boolean useGpu) {
			this.useGpu = useGpu;
			return this;
		}

		public Builder samplingParams(SamplingParams samplingParams) {
			if (samplingParams == null) {
				throw new IllegalArgumentException("samplingParams must not be null");
			}
			this.samplingParams = samplingParams;
			return this;
		}

		/**
		 * Override the system prompt injected at the start of every request. Pass
		 * {@code null} or blank to disable the system prompt.
		 */
		public Builder systemPrompt(String systemPrompt) {
			this.systemPrompt = systemPrompt;
			return this;
		}

		/**
		 * Apply a trained LoRA adapter at inference time (equivalent to
		 * {@code ./juno local --lora-play <path>}).
		 *
		 * <p>
		 * The adapter file must have been produced by
		 * {@link cab.ml.juno.player.LoraTrainer#save()} or the {@code /save} REPL
		 * command. When set, the pipeline is routed through
		 * {@link cab.ml.juno.node.LoraTrainableHandler} in inference-only mode (no
		 * optimizer attached).
		 *
		 * @param loraPath path to the {@code .lora} checkpoint file; {@code null}
		 *                 disables LoRA (base model only)
		 */
		public Builder loraPlay(Path loraPath) {
			this.loraPlayPath = loraPath;
			return this;
		}

		/**
		 * Loads the GGUF model and wires the local inference pipeline.
		 *
		 * @throws IOException if the model file cannot be read
		 */
		public LocalChat build() throws IOException {
			System.setProperty("juno.byteOrder", "BE");

			LlamaConfig config;
			Tokenizer tokenizer;
			try (GgufReader reader = GgufReader.open(modelPath)) {
				config = LlamaConfig.from(reader);
				tokenizer = GgufTokenizer.load(reader);
			}

			long vramPerLayerBytes = estimateVramPerLayer(config.hiddenDim());
			long nodeVramBytes = config.numLayers() * vramPerLayerBytes * 2;

			List<NodeDescriptor> nodes = new ArrayList<>();
			for (int i = 0; i < nodeCount; i++) {
				nodes.add(new NodeDescriptor("node-" + i, "localhost", 9092 + i, nodeVramBytes, nodeVramBytes,
						NodeStatus.READY, 1.0, Instant.now(), Instant.now()));
			}

			ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayerBytes, nodes);

			GpuContext gpuCtx = resolveGpuContext(useGpu);
			MatVec sharedBackend = (gpuCtx != null) ? gpuCtx.createMatVec()
					: ForwardPassHandlerLoader.selectBackend();

			LoraAdapterSet adapters = loadAdapters(loraPlayPath);

			List<ForwardPassHandler> handlers = new ArrayList<>();
			for (var assignment : shardMap.assignments()) {
				ShardContext ctx = ShardContext.from(assignment, config.vocabSize(), config.hiddenDim(),
						config.numHeads());
				handlers.add(ForwardPassHandlerLoader.load(modelPath, ctx, sharedBackend, adapters));
			}

			LocalInferencePipeline pipeline = LocalInferencePipeline.from(shardMap, new ArrayList<>(handlers),
					config.vocabSize(), config.hiddenDim(), config.numHeads());

			KVCacheManager kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));

			GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

			String modelType = ChatModelType.fromPath(modelPath.toString());

			return new LocalChat(loop, modelType, samplingParams, List.copyOf(handlers), gpuCtx, systemPrompt);
		}

		private static LoraAdapterSet loadAdapters(Path loraPath) throws IOException {
			if (loraPath == null) {
				return null;
			}
			return LoraAdapterSet.load(loraPath);
		}

		private static long estimateVramPerLayer(int hiddenDim) {
			return (long) (4L * hiddenDim * hiddenDim * 2.0);
		}

		private static GpuContext resolveGpuContext(boolean useGpu) {
			if (!useGpu || !CudaAvailability.isAvailable()) {
				return null;
			}
			int device = Math.max(0, Integer.getInteger("juno.cuda.device", 0));
			if (device >= CudaAvailability.deviceCount()) {
				return null;
			}
			return GpuContext.shared(device);
		}
	}
}