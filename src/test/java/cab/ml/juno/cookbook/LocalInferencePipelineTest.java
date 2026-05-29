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

import cab.ml.juno.sampler.SamplingParams;

/**
 * In-process inference tests for {@link JunoLocalRepl}.
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
 * <li>{@link JunoLocalRepl#runInteractive} reads from InputStream and writes to
 * PrintStream correctly.</li>
 * <li>Blank input is rejected with {@link IllegalArgumentException}.</li>
 * </ul>
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LocalInferencePipelineTest {

	private static final String MODEL_PATH = "/home/robocop/dev/models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf";

	// Shared repl instance — loaded once, reused across tests for speed.
	// Each test that mutates history calls resetHistory() before and/or after.
	private static LocalInferencePipelineExample repl;

	@BeforeAll
	static void loadModel() throws Exception {

		String modelPath = MODEL_PATH;
		assumeTrue(modelPath != null && !modelPath.isBlank(),
				"Skipping: set -D" + MODEL_PATH + "=/path/to/model.gguf to run inference tests.");
		assumeTrue(Files.exists(Path.of(modelPath)), "Skipping: model file not found at " + modelPath);

		repl = LocalInferencePipelineExample.builder(Path.of(modelPath)).nodeCount(1) // single shard — smallest memory
																						// footprint for tests
				.useGpu(false) // CPU-only — no CUDA required on CI
				.samplingParams(SamplingParams.defaults().withMaxTokens(64).withTemperature(0.7f)).build();
	}

	@AfterAll
	static void closeModel() {
		if (repl != null) {
			repl.close();
		}
	}

	// ── Tests ─────────────────────────────────────────────────────────────────

	@Test
	@Order(1)
	@DisplayName("single turn returns a non-empty reply")
	void singleTurnReturnsNonEmptyReply() {
		repl.resetHistory();
		String reply = repl.chat("Say exactly the word: hello");
		assertThat(reply).isNotBlank();
	}

	@Test
	@Order(2)
	@DisplayName("multi-turn: model recalls a fact stated in a prior turn")
	void multiTurnContextRecall() {
		repl.resetHistory();

		// Turn 1 — plant the fact.
		repl.chat("My name is Maximilian. Please remember that.");

		// Turn 2 — retrieve it without restating it.
		String reply = repl.chat("What is my name? Reply with just the name and nothing else.");

		assertThat(reply.toLowerCase()).contains("maximilian");
	}

	@Test
	@Order(3)
	@DisplayName("resetHistory clears context — model no longer recalls prior turns")
	void resetHistoryClearsContext() {
		repl.resetHistory();

		// Plant the fact in one session.
		repl.chat("My secret code is ZEPHYR42. Remember it.");
		repl.chat("Confirm you know my code."); // give it a chance to confirm

		// Reset — everything above is gone.
		repl.resetHistory();

		// Fresh session — the model must not recall the code.
		String reply = repl.chat("Do you know my secret code? " + "If you have no idea, just say you do not know.");

		assertThat(reply.toLowerCase()).doesNotContain("zephyr42");
	}

	@Test
	@Order(4)
	@DisplayName("deterministic mode: same prompt produces the same reply twice")
	void deterministicModeProducesSameReply() throws Exception {
		Path modelPath = Path.of(MODEL_PATH);
		SamplingParams deterministic = SamplingParams.deterministic().withMaxTokens(32);

		try (LocalInferencePipelineExample lip = LocalInferencePipelineExample.builder(modelPath).nodeCount(1)
				.useGpu(false).samplingParams(deterministic).build()) {

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
		repl.resetHistory();

		repl.chat("Turn one.");
		repl.chat("Turn two.");

		// 2 user messages + 2 assistant replies = 4 entries
		assertThat(repl.history()).hasSize(4);
		assertThat(repl.history().get(0).isUser()).isTrue();
		assertThat(repl.history().get(1).isAssistant()).isTrue();
		assertThat(repl.history().get(2).isUser()).isTrue();
		assertThat(repl.history().get(3).isAssistant()).isTrue();
	}

	@Test
	@Order(6)
	@DisplayName("resetHistory produces a new session ID each time")
	void resetHistoryChangesSessionId() {
		repl.resetHistory();
		String first = repl.sessionId();
		repl.resetHistory();
		String second = repl.sessionId();
		assertThat(first).isNotEqualTo(second);
	}

	@Test
	@Order(7)
	@DisplayName("runInteractive reads lines from InputStream and writes replies to PrintStream")
	void runInteractiveReadsAndWritesStreams() {
		repl.resetHistory();

		String input = "What is 1 + 1?\nexit\n";
		ByteArrayInputStream in = new ByteArrayInputStream(input.getBytes());
		ByteArrayOutputStream captured = new ByteArrayOutputStream();
		PrintStream out = new PrintStream(captured);

		repl.runInteractive(in, out);

		String output = captured.toString();
		assertThat(output).contains("bot>");
		assertThat(output).isNotBlank();
	}

	@Test
	@Order(8)
	@DisplayName("chat rejects blank input")
	void chatRejectsBlankInput() {
		assertThatThrownBy(() -> repl.chat("   ")).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("userText must not be blank");
	}

	@Test
	@Order(9)
	@DisplayName("chat rejects null input")
	void chatRejectsNullInput() {
		assertThatThrownBy(() -> repl.chat(null)).isInstanceOf(IllegalArgumentException.class);
	}

	// ── Builder validation ─────────────────────────────────────────────────

	@Test
	@Order(10)
	@DisplayName("builder rejects null modelPath")
	void builderRejectsNullModelPath() {
		assertThatThrownBy(() -> LocalInferencePipelineExample.builder(null))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@Order(11)
	@DisplayName("builder rejects null samplingParams")
	void builderRejectsNullSamplingParams() {
		assertThatThrownBy(() -> LocalInferencePipelineExample.builder(Path.of("/tmp/model.gguf")).samplingParams(null))
				.isInstanceOf(IllegalArgumentException.class);
	}
}