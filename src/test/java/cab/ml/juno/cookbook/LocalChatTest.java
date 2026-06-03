package cab.ml.juno.cookbook;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
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
 * </ul>
 */

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LocalChatTest {

	private static final String MODEL_PATH = "/home/robocop/dev/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf";

	private static LocalChat lc;

	@BeforeAll
	static void buildPipline() throws Exception {
		lc = LocalChat.builder(Path.of(MODEL_PATH)).nodeCount(1).useGpu(false)
				.samplingParams(SamplingParams.defaults().withMaxTokens(64).withTemperature(0.7f)).build();
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
		assertThat(reply.toLowerCase()).contains("viktor");
	}

	@Test
	@Order(3)
	@DisplayName("resetHistory clears context — model no longer recalls prior turns")
	void resetHistoryClearsContext() {
		lc.resetHistory();
		lc.chat("My secret code is ZEPHYR42. Remember it.");
		String reply = lc.chat("Confirm you know my code.");
		assertThat(reply.toUpperCase()).contains("ZEPHYR42");
		lc.resetHistory();
		String reply2 = lc.chat("Do you know my secret code? If you have no idea, just say you do not know.");
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
}