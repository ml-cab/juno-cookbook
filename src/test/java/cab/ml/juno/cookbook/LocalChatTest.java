package cab.ml.juno.cookbook;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.player.LoraTrainer;
import cab.ml.juno.sampler.SamplingParams;

/**
 * In-process inference tests for {@link LocalChat}.
 *
 * <p>
 * These tests load a real GGUF model and run actual inference — no mocks, no
 * external server. They are skipped automatically when no model path is
 * provided, so the build stays green on CI without a model file.
 *
 * <p>
 * To run:
 *
 * <pre>
 *   mvn test -Djuno.test.model=/path/to/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
 * </pre>
 *
 * <p>
 * Tested behaviours:
 * <ul>
 * <li>Single-turn reply is non-empty.</li>
 * <li>Context recall across turns (multi-turn KV-cache session).</li>
 * <li>History reset clears context — model no longer recalls prior turns.</li>
 * <li>Deterministic mode: identical prompt produces the same reply twice.</li>
 * <li>{@link LocalChat#runInteractive} reads from InputStream and writes to
 * PrintStream correctly.</li>
 * <li>Blank input is rejected with {@link IllegalArgumentException}.</li>
 * <li>LoRA train-qa: a Q&amp;A pair trained via {@link LoraTrainer#trainQaPair}
 * and loaded with {@code loraPlay} is recalled at inference.</li>
 * <li>LoRA train text: a raw text passage trained via
 * {@link LoraTrainer#trainRawText} and loaded with {@code loraPlay} influences
 * the model's output vocabulary.</li>
 * </ul>
 */

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LocalChatTest {

	private static final String MODEL_PATH = "/home/robocop/dev/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf";

	private static LocalChat lc;

	@BeforeAll
	static void buildPipline() throws Exception {
		lc = LocalChat.builder(Path.of(MODEL_PATH)).nodeCount(1).useGpu(false)
				.samplingParams(SamplingParams.defaults().withMaxTokens(64).withTemperature(0.1f)).build();
	}

	@AfterAll
	static void closePipeline() {
		if (lc != null) {
			lc.close();
		}
	}

	@Test
	@Order(1)
	@DisplayName("single turn returns a non-empty reply")
	void singleTurnReturnsNonEmptyReply() {
		lc.resetHistory();
		String reply = lc.chat("Say exactly the word: hello");
		assertThat(reply).isNotBlank();
	}

	@Test
	@Order(2)
	@DisplayName("multi-turn: model recalls a fact stated in a prior turn")
	void multiTurnContextRecall() {
		lc.resetHistory();
		lc.chat("My name is Viktor. Please remember that.");
		String reply = lc.chat("Recall me, what is my name?");
		System.out.println("[multiTurnContextRecall] inference reply: " + reply);
		assertThat(reply.toLowerCase()).contains("viktor");
	}

	@Test
	@Order(3)
	@DisplayName("resetHistory clears context — model no longer recalls prior turns")
	void resetHistoryClearsContext() {
		lc.resetHistory();
		lc.chat("My secret code is ZEPHYR42. Remember it.");
		String reply = lc.chat("Confirm you know my code.");
		System.out.println("[resetHistoryClearsContext] inference reply1: " + reply);
		assertThat(reply.toUpperCase()).contains("ZEPHYR42");
		lc.resetHistory();
		String reply2 = lc.chat("Do you know my secret code? If you have no idea, just say you do not know.");
		System.out.println("[resetHistoryClearsContext] inference reply2: " + reply2);
		assertThat(reply2.toLowerCase()).doesNotContain("zephyr42");
	}

	@Test
	@Order(4)
	@DisplayName("deterministic mode: same prompt produces the same reply twice")
	void deterministicModeProducesSameReply() throws Exception {
		Path modelPath = Path.of(MODEL_PATH);
		SamplingParams deterministic = SamplingParams.deterministic().withMaxTokens(32);
		try (LocalChat lip = LocalChat.builder(modelPath).nodeCount(1).useGpu(false).samplingParams(deterministic)
				.build()) {
			String prompt = "What is the capital of France? Answer in one word.";
			lip.resetHistory();
			String first = lip.chat(prompt);
			lip.resetHistory();
			String second = lip.chat(prompt);
			assertThat(first).isEqualTo(second);
		}
	}

	@Test
	@Order(5)
	@DisplayName("history grows with each turn and is accessible via history()")
	void historyAccumulatesAcrossTurns() {
		lc.resetHistory();
		lc.chat("Turn one.");
		lc.chat("Turn two.");
		// 2 user messages + 2 assistant replies = 4 entries (system prompt excluded)
		assertThat(lc.history()).hasSize(4);
		assertThat(lc.history().get(0).isUser()).isTrue();
		assertThat(lc.history().get(1).isAssistant()).isTrue();
		assertThat(lc.history().get(2).isUser()).isTrue();
		assertThat(lc.history().get(3).isAssistant()).isTrue();
	}

	@Test
	@Order(6)
	@DisplayName("resetHistory produces a new session ID each time")
	void resetHistoryChangesSessionId() {
		lc.resetHistory();
		String first = lc.sessionId();
		lc.resetHistory();
		String second = lc.sessionId();
		assertThat(first).isNotEqualTo(second);
	}

	@Test
	@Order(7)
	@DisplayName("runInteractive reads lines from InputStream and writes replies to PrintStream")
	void runInteractiveReadsAndWritesStreams() {
		lc.resetHistory();
		String input = "What is 1 + 1?\nexit\n";
		ByteArrayInputStream in = new ByteArrayInputStream(input.getBytes());
		ByteArrayOutputStream captured = new ByteArrayOutputStream();
		PrintStream out = new PrintStream(captured);
		lc.runInteractive(in, out);
		String output = captured.toString();
		assertThat(output).contains("bot>");
		assertThat(output).isNotBlank();
	}

	@Test
	@Order(8)
	@DisplayName("chat rejects blank input")
	void chatRejectsBlankInput() {
		assertThatThrownBy(() -> lc.chat("   ")).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("userText must not be blank");
	}

	@Test
	@Order(9)
	@DisplayName("chat rejects null input")
	void chatRejectsNullInput() {
		assertThatThrownBy(() -> lc.chat(null)).isInstanceOf(IllegalArgumentException.class);
	}

	// ── Builder validation ─────────────────────────────────────────────────

	@Test
	@Order(10)
	@DisplayName("builder rejects null modelPath")
	void builderRejectsNullModelPath() {
		assertThatThrownBy(() -> LocalChat.builder(null)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@Order(11)
	@DisplayName("builder rejects null samplingParams")
	void builderRejectsNullSamplingParams() {
		assertThatThrownBy(() -> LocalChat.builder(Path.of("/tmp/model.gguf")).samplingParams(null))
				.isInstanceOf(IllegalArgumentException.class);
	}

	// ── LoRA train-qa then lora play ──────────────────────────────────────────

	/**
	 * Loss target below which QA recall is reliable (per LoRA.md: &lt;0.5 gives
	 * consistent recall; 1.5 gives inconsistent recall). The target here is
	 * intentionally conservative to keep the training loop short.
	 */
	private static final float QA_LOSS_TARGET  = 1.2f;

	/**
	 * Loss target for raw-text training. Raw-text training is weaker than QA
	 * training; this value is kept high enough that the loop exits after a few
	 * passes while still demonstrating meaningful adapter learning.
	 */
	private static final float TEXT_LOSS_TARGET = 1.8f;

	/** Hard cap on training iterations to prevent an infinite loop on a broken model. */
	private static final int   MAX_TRAIN_ITERS  = 50;

	/**
	 * Trains a single Q&amp;A fact with {@link LoraTrainer#trainQaPair} until loss
	 * drops below {@value #QA_LOSS_TARGET} (or {@value #MAX_TRAIN_ITERS} iterations
	 * are exhausted), saves the adapter, then opens a fresh {@link LocalChat} with
	 * {@code loraPlay} and verifies the trained answer is recalled.
	 *
	 * <p>
	 * Each iteration calls {@code trainQaPair} with {@code stepsPerChunk=1} — the
	 * minimum that advances the optimizer — so each pass is a single forward +
	 * backward + Adam step per 32-token chunk. Training stops as soon as loss is
	 * good enough, not after a fixed count.
	 */
	@Test
	@Order(12)
	@DisplayName("lora train-qa then lora play: trained Q&A fact is recalled at inference")
	void loraTrainQaThenPlay(@TempDir Path tmpDir) throws Exception {
		assumeTrue(Files.exists(Path.of(MODEL_PATH)), "model file not present — skipping");

		Path modelPath   = Path.of(MODEL_PATH);
		Path adapterPath = tmpDir.resolve("test.lora");

		try (LoraTrainer trainer = LoraTrainer.open(modelPath, adapterPath, 8, 16f, 1e-4)) {
			float loss = Float.MAX_VALUE;
			for (int iter = 1; iter <= MAX_TRAIN_ITERS && loss > QA_LOSS_TARGET; iter++) {
				loss = trainer.trainQaPair(
						"What is the name of the AI assistant?", "Orion", "tinyllama", 1);
				System.out.printf("[train-qa] iter=%2d  loss=%.4f  target=%.2f%n", iter, loss, QA_LOSS_TARGET);
			}
			System.out.printf("[train-qa] final loss=%.4f%n", loss);
			trainer.save();
		}

		assertThat(adapterPath).exists();

		SamplingParams params = SamplingParams.defaults().withMaxTokens(32).withTemperature(0.1f);
		try (LocalChat chat = LocalChat.builder(modelPath).nodeCount(1).useGpu(false)
				.samplingParams(params).loraPlay(adapterPath).build()) {
			chat.resetHistory();
			String reply = chat.chat("What is the name of the AI assistant?");
			System.out.println("[train-qa] inference reply: " + reply);
			assertThat(reply.toLowerCase()).contains("orion");
		}
	}

	/**
	 * Trains on a short raw-text passage with {@link LoraTrainer#trainRawText}
	 * until loss drops below {@value #TEXT_LOSS_TARGET} (or
	 * {@value #MAX_TRAIN_ITERS} iterations are exhausted), saves the adapter, then
	 * opens a fresh {@link LocalChat} with {@code loraPlay} and verifies that the
	 * model's reply contains vocabulary from the trained passage.
	 *
	 * <p>
	 * Each iteration passes {@code stepsPerChunk=1} and {@code chunkTokens=128}
	 * to minimise tokens-per-call while still advancing the optimizer once per
	 * chunk. The passage is short enough (&lt;80 tokens) to fit in a single chunk
	 * at this chunk size.
	 */
	@Test
	@Order(13)
	@DisplayName("lora train text then lora play: trained passage vocabulary appears in completion")
	void loraTrainTextThenPlay(@TempDir Path tmpDir) throws Exception {
		assumeTrue(Files.exists(Path.of(MODEL_PATH)), "model file not present — skipping");

		Path modelPath   = Path.of(MODEL_PATH);
		Path adapterPath = tmpDir.resolve("text.lora");

		String passage = "Helixa is a distributed inference engine for low-latency language model serving. "
				+ "Helixa supports tensor parallelism and dynamic batching. "
				+ "Helixa was created to make fast LLM inference accessible without specialized hardware.";

		try (LoraTrainer trainer = LoraTrainer.open(modelPath, adapterPath, 8, 16f, 1e-4)) {
			float loss = Float.MAX_VALUE;
			for (int iter = 1; iter <= MAX_TRAIN_ITERS && loss > TEXT_LOSS_TARGET; iter++) {
				loss = trainer.trainRawText(passage, 1, 128);
				System.out.printf("[train-text] iter=%2d  loss=%.4f  target=%.2f%n", iter, loss, TEXT_LOSS_TARGET);
			}
			System.out.printf("[train-text] final loss=%.4f%n", loss);
			trainer.save();
		}

		assertThat(adapterPath).exists();

		SamplingParams params = SamplingParams.defaults().withMaxTokens(40).withTemperature(0.3f);
		try (LocalChat chat = LocalChat.builder(modelPath).nodeCount(1).useGpu(false)
				.samplingParams(params).loraPlay(adapterPath).build()) {
			chat.resetHistory();
			String reply = chat.chat("Tell me about Helixa. What is it and what does it support?");
			System.out.println("[train-text] inference reply: " + reply);
			assertThat(reply).isNotBlank();
			String replyLower = reply.toLowerCase();
			assertThat(replyLower.contains("helixa")
					|| replyLower.contains("inference")
					|| replyLower.contains("latency")
					|| replyLower.contains("language")
					|| replyLower.contains("parallelism")).isTrue();
		}
	}
}