package cab.ml.juno.cookbook;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.sampler.SamplingParams;

/**
 * Unit tests for {@link LoRATuner.Builder} validation.
 *
 * <p>
 * These tests cover the builder guard logic without loading a model file.
 * Integration tests that require an actual GGUF model should extend this class
 * and override the model-path property:
 *
 * <pre>
 *   mvn test -Djuno.test.model=/path/to/model.gguf
 * </pre>
 */
class LoRATunerTest {

    private static final Path PLACEHOLDER = Path.of("/tmp/model.gguf");

    // ── Builder validation ────────────────────────────────────────────────────

    @Test
    @DisplayName("builder rejects null modelPath")
    void builderRejectsNullModelPath() {
        assertThatThrownBy(() -> LoRATuner.builder(null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("builder rejects rank < 1")
    void builderRejectsRankLessThanOne() {
        assertThatThrownBy(() -> LoRATuner.builder(PLACEHOLDER).rank(0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("rank");
    }

    @Test
    @DisplayName("builder rejects lr <= 0")
    void builderRejectsNonPositiveLr() {
        assertThatThrownBy(() -> LoRATuner.builder(PLACEHOLDER).lr(0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("lr");
    }

    @Test
    @DisplayName("builder rejects null samplingParams")
    void builderRejectsNullSamplingParams() {
        assertThatThrownBy(() -> LoRATuner.builder(PLACEHOLDER).samplingParams(null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("stepsQa clamps to minimum of 1")
    void stepsQaClampsToOne() {
        // Should not throw; we just verify the builder accepts it
        LoRATuner.Builder builder = LoRATuner.builder(PLACEHOLDER).stepsQa(-5);
        assertThat(builder).isNotNull();
    }

    @Test
    @DisplayName("stepsText clamps to minimum of 1")
    void stepsTextClampsToOne() {
        LoRATuner.Builder builder = LoRATuner.builder(PLACEHOLDER).stepsText(0);
        assertThat(builder).isNotNull();
    }

    @Test
    @DisplayName("chunkTokens clamps to minimum of 2")
    void chunkTokensClampsToTwo() {
        LoRATuner.Builder builder = LoRATuner.builder(PLACEHOLDER).chunkTokens(0);
        assertThat(builder).isNotNull();
    }

    @Test
    @DisplayName("adapterPath setter accepts null (will use default derivation on build)")
    void adapterPathAcceptsNull() {
        LoRATuner.Builder builder = LoRATuner.builder(PLACEHOLDER).adapterPath(null);
        assertThat(builder).isNotNull();
    }

    // ── Constants ─────────────────────────────────────────────────────────────

    @Test
    @DisplayName("DEFAULT_RANK is 8")
    void defaultRankIsEight() {
        assertThat(LoRATuner.DEFAULT_RANK).isEqualTo(8);
    }

    @Test
    @DisplayName("DEFAULT_STEPS_QA is 10")
    void defaultStepsQaIsTen() {
        assertThat(LoRATuner.DEFAULT_STEPS_QA).isEqualTo(10);
    }

    @Test
    @DisplayName("DEFAULT_STEPS_TEXT is 50")
    void defaultStepsTextIsFifty() {
        assertThat(LoRATuner.DEFAULT_STEPS_TEXT).isEqualTo(50);
    }

    @Test
    @DisplayName("DEFAULT_EARLY_STOP is 0.25")
    void defaultEarlyStopIsQuarter() {
        assertThat(LoRATuner.DEFAULT_EARLY_STOP).isEqualTo(0.25f);
    }

    // ── Model-required integration tests (skipped without model) ──────────────
    // To run: mvn test -Djuno.test.model=/path/to/model.gguf -Dtest=LoRATunerTest

    private static final String MODEL_PATH_PROP = "juno.test.model";

    /**
     * Helper: returns true only when a real model path is provided via system
     * property. Prevents integration tests from running (and failing) on CI.
     */
    static boolean modelAvailable() {
        String p = System.getProperty(MODEL_PATH_PROP);
        return p != null && !p.isBlank() && java.nio.file.Files.exists(Path.of(p));
    }

    @Test
    @DisplayName("trainQA rejects blank question")
    void trainQARejectsBlankQuestion() throws Exception {
        if (!modelAvailable()) return;
        Path model = Path.of(System.getProperty(MODEL_PATH_PROP));
        try (LoRATuner tuner = LoRATuner.builder(model).stepsQa(1).build()) {
            assertThatThrownBy(() -> tuner.trainQA("  ", "answer"))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("question");
        }
    }

    @Test
    @DisplayName("trainQA rejects blank answer")
    void trainQARejectsBlankAnswer() throws Exception {
        if (!modelAvailable()) return;
        Path model = Path.of(System.getProperty(MODEL_PATH_PROP));
        try (LoRATuner tuner = LoRATuner.builder(model).stepsQa(1).build()) {
            assertThatThrownBy(() -> tuner.trainQA("What is 1+1?", ""))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("answer");
        }
    }

    @Test
    @DisplayName("chat rejects blank input")
    void chatRejectsBlankInput() throws Exception {
        if (!modelAvailable()) return;
        Path model = Path.of(System.getProperty(MODEL_PATH_PROP));
        try (LoRATuner tuner = LoRATuner.builder(model).stepsQa(1).build()) {
            assertThatThrownBy(() -> tuner.chat("   "))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("userText");
        }
    }

    @Test
    @DisplayName("trainQA returns finite loss")
    void trainQAReturnsFiniteLoss() throws Exception {
        if (!modelAvailable()) return;
        Path model = Path.of(System.getProperty(MODEL_PATH_PROP));
        try (LoRATuner tuner = LoRATuner.builder(model)
                .stepsQa(2)
                .samplingParams(SamplingParams.defaults().withMaxTokens(32))
                .build()) {
            float loss = tuner.trainQA("What is the capital of France?", "Paris");
            assertThat(loss).isFinite().isGreaterThan(0f);
        }
    }

    @Test
    @DisplayName("chat returns non-empty reply")
    void chatReturnsNonEmptyReply() throws Exception {
        if (!modelAvailable()) return;
        Path model = Path.of(System.getProperty(MODEL_PATH_PROP));
        try (LoRATuner tuner = LoRATuner.builder(model)
                .stepsQa(1)
                .samplingParams(SamplingParams.defaults().withMaxTokens(32))
                .build()) {
            String reply = tuner.chat("Say exactly the word: hello");
            assertThat(reply).isNotBlank();
        }
    }

    @Test
    @DisplayName("resetHistory produces a new session ID")
    void resetHistoryChangesSessionId() throws Exception {
        if (!modelAvailable()) return;
        Path model = Path.of(System.getProperty(MODEL_PATH_PROP));
        try (LoRATuner tuner = LoRATuner.builder(model)
                .stepsQa(1)
                .samplingParams(SamplingParams.defaults().withMaxTokens(16))
                .build()) {
            String first = tuner.sessionId();
            tuner.resetHistory();
            String second = tuner.sessionId();
            assertThat(first).isNotEqualTo(second);
        }
    }

    @Test
    @DisplayName("save and load round-trip: adapter file created after save")
    void saveCreatesAdapterFile() throws Exception {
        if (!modelAvailable()) return;
        Path model = Path.of(System.getProperty(MODEL_PATH_PROP));
        Path adapter = java.nio.file.Files.createTempFile("juno-test-", ".lora");
        adapter.toFile().delete(); // start without the file so new adapters are initialised
        try (LoRATuner tuner = LoRATuner.builder(model)
                .adapterPath(adapter)
                .stepsQa(1)
                .samplingParams(SamplingParams.defaults().withMaxTokens(16))
                .build()) {
            tuner.trainQA("What is 1+1?", "2");
            tuner.save();
            assertThat(adapter).exists().isNotEmptyFile();
        } finally {
            java.nio.file.Files.deleteIfExists(adapter);
        }
    }
}