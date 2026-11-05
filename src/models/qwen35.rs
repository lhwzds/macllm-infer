use anyhow::{Result, bail, ensure};

use crate::{
    agent::{AgentPrompt, AgentTemplateProfile},
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatTemplateOptions},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::{GenericPaths, GgufPromptTokens, JinjaChatFallback, ModelAdapter, jinja_fallback},
};

// The prompt renderers live in the chat group so that the serving and
// benchmark layers can use them without depending on the model adapters.
pub use crate::chat_templates::qwen35::{
    OFFICIAL_CHAT_TEMPLATE_LEN, OFFICIAL_CHAT_TEMPLATE_SHA256, OFFICIAL_GGUF_CHAT_TEMPLATE_LEN,
    OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256, Qwen35ChatTemplate, UNSLOTH_GGUF_CHAT_TEMPLATE_LEN,
    UNSLOTH_GGUF_CHAT_TEMPLATE_SHA256, classify_chat_template, format_qwen35_agent_prompt,
    format_qwen35_chat_messages, format_qwen35_plain,
};

// The shared decoder and the adapters read the same weight-name table, so it
// lives in the architecture module and is re-exported here.
pub(crate) use crate::architecture::QWEN35_WEIGHT_NAMES;

/// Adapter for the dense Qwen3.5 text trunk stored in canonical llama.cpp
/// GGUF form.
///
/// Exact template identities are recognized by digest. The two newer
/// identities support text and native tool history; the older GGUF identity
/// remains limited to its verified plain-text subset. Multimodal input is not
/// supported.
#[derive(Debug, Clone)]
pub struct Qwen35Adapter {
    spec: DecoderSpec,
    chat_template: Option<Qwen35ChatTemplate>,
    /// The embedded template of a checkpoint that is not one of the pinned
    /// programs, compiled for the generic Jinja renderer.
    jinja: Option<JinjaChatFallback>,
}

impl Qwen35Adapter {
    /// Construct an adapter from the metadata and tensors already validated by
    /// the GGUF loader.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "qwen35",
            "expected a qwen35 decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
            jinja: None,
        })
    }
}

impl ModelAdapter for Qwen35Adapter {
    fn family(&self) -> &'static str {
        "qwen35"
    }

    fn has_pinned_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    fn runtime_spec(&self) -> &RuntimeSpec {
        &self.spec.runtime
    }

    fn decoder_spec(&self) -> Option<&DecoderSpec> {
        Some(&self.spec)
    }

    fn build_model(
        &self,
        weights: &dyn WeightSource,
    ) -> candle_core::Result<Box<dyn crate::architecture::CausalLm>> {
        Ok(Box::new(DecoderModel::load_with_weight_names(
            self.spec.clone(),
            weights,
            &QWEN35_WEIGHT_NAMES,
        )?))
    }

    fn attach_prompt_tokens(&mut self, tokens: GgufPromptTokens<'_>) -> Result<()> {
        // The chat path and the agent path are pinned separately: a
        // checkpoint with a byte-pinned chat template but no native agent
        // profile still renders its tool calls through the generic
        // renderer.
        self.jinja = jinja_fallback(
            self.family(),
            GenericPaths {
                chat: !self.has_pinned_chat_template(),
                agent: self.agent_template_profile().is_none(),
            },
            tokens,
        );
        Ok(())
    }

    fn format_chat_messages(&self, messages: &[ChatMessage]) -> Result<String> {
        self.format_chat_messages_with(messages, ChatTemplateOptions::default())
    }

    fn format_chat_messages_with(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        match self.chat_template {
            Some(Qwen35ChatTemplate::OfficialTextSubset) => {
                options.ensure_default(self.family())?;
                format_qwen35_chat_messages(messages)
            }
            Some(template @ (Qwen35ChatTemplate::Official | Qwen35ChatTemplate::Unsloth)) => {
                format_qwen35_plain(messages, template == Qwen35ChatTemplate::Official, options)
            }
            None => match &self.jinja {
                Some(jinja) => jinja.format_chat_messages(messages, options),
                None => bail!(
                    "unsupported or missing Qwen3.5 chat template; use `run --raw-prompt` or a pinned Qwen3.5 template"
                ),
            },
        }
    }

    fn chat_template_defaults(&self) -> ChatTemplateOptions {
        ChatTemplateOptions {
            enable_thinking: match self.chat_template {
                Some(Qwen35ChatTemplate::Official) => Some(true),
                Some(Qwen35ChatTemplate::Unsloth) => Some(false),
                Some(Qwen35ChatTemplate::OfficialTextSubset) | None => None,
            },
        }
    }

    fn tool_call_format(&self) -> Option<macllm_chat::tool_format::ToolCallFormat> {
        self.jinja
            .as_ref()
            .and_then(JinjaChatFallback::tool_call_format)
    }

    fn agent_template_profile(&self) -> Option<AgentTemplateProfile> {
        match self.chat_template {
            Some(Qwen35ChatTemplate::Official) => Some(AgentTemplateProfile::Qwen35Official),
            Some(Qwen35ChatTemplate::Unsloth) => Some(AgentTemplateProfile::Qwen35Unsloth),
            _ => None,
        }
    }
    fn format_agent_prompt(&self, prompt: &AgentPrompt) -> Result<String> {
        self.format_agent_prompt_with(prompt, ChatTemplateOptions::default())
    }
    fn format_agent_prompt_with(
        &self,
        prompt: &AgentPrompt,
        options: ChatTemplateOptions,
    ) -> Result<String> {
        if self.agent_template_profile().is_none()
            && let Some(jinja) = &self.jinja
        {
            return jinja.format_agent_prompt(prompt, options);
        }
        let profile = self
            .agent_template_profile()
            .ok_or_else(|| anyhow::anyhow!("unsupported Qwen3.5 native agent template"))?;
        format_qwen35_agent_prompt(profile, prompt, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::io::Read;
    #[cfg(unix)]
    use std::os::fd::AsRawFd;
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;
    #[cfg(unix)]
    use std::process::{Child, Command, ExitStatus, Stdio};
    #[cfg(unix)]
    use std::sync::atomic::{AtomicBool, Ordering};
    #[cfg(unix)]
    use std::sync::{Arc, mpsc};
    #[cfg(unix)]
    use std::thread::JoinHandle;
    #[cfg(unix)]
    use std::time::{Duration, Instant};

    /// The raw bytes that one pipe reader keeps.
    #[cfg(unix)]
    const MAX_PIPE_BYTES: usize = 1024 * 1024;

    /// The grace that a child which ignores `SIGTERM` gets before `SIGKILL`.
    #[cfg(unix)]
    const TERM_GRACE_MS: u64 = 200;

    /// The bound of the reap after `SIGKILL`, and of the join of a reader.
    #[cfg(unix)]
    const STOP_SECS: u64 = 1;

    /// The bound of a child that reruns one test of this binary.
    #[cfg(unix)]
    const CHILD_TIMEOUT_SECS: u64 = 60;

    /// The bound of a call whose readers end on the drain deadline: the call
    /// must not wait for the deadline of the command.
    #[cfg(unix)]
    const DRAIN_CALL_BOUND: Duration = Duration::from_secs(10);

    /// The error of a timed-out child, carrying its process group.
    #[cfg(unix)]
    #[derive(Debug)]
    struct ChildTimeout(i32);

    #[cfg(unix)]
    impl std::fmt::Display for ChildTimeout {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("child timed out")
        }
    }

    #[cfg(unix)]
    impl std::error::Error for ChildTimeout {}

    /// What one pipe reader reports when it ends: the text it kept, and
    /// whether it reached the end of its pipe.
    #[cfg(unix)]
    struct PipeRead {
        text: String,
        eof: bool,
    }

    /// The child of a test and the readers of its pipes. The guard is the only
    /// reaper: it reaps through `Child::try_wait` alone, never signalling a
    /// child that `try_wait` has reaped.
    #[cfg(unix)]
    struct ChildGuard {
        child: Child,
        /// The process group of the child, which the child leads.
        pgid: i32,
        /// Whether `try_wait` has reaped the child.
        reaped: bool,
        /// The instant at which `try_wait` reaped the child.
        reaped_at: Option<Instant>,
        stop_readers: Arc<AtomicBool>,
        readers: Vec<(JoinHandle<()>, mpsc::Receiver<PipeRead>)>,
    }

    #[cfg(unix)]
    impl ChildGuard {
        /// Spawn the command, and start the readers of its pipes.
        fn spawn(command: &mut Command) -> Result<Self> {
            /// Read one pipe in a thread: nonblocking, capped, decoded once.
            /// The reader ends on the end of its pipe, on the stop flag, or on
            /// the drain deadline: never on the deadline of the command.
            fn read_pipe(
                mut pipe: impl Read + AsRawFd + Send + 'static,
                stop: Arc<AtomicBool>,
            ) -> Result<(JoinHandle<()>, mpsc::Receiver<PipeRead>)> {
                let fd = pipe.as_raw_fd();
                let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
                ensure!(flags >= 0, "could not read the flags of a pipe");
                let set = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
                ensure!(set >= 0, "could not make a pipe nonblocking");
                let (tx, rx) = mpsc::channel();
                let handle = std::thread::spawn(move || {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut dropped = false;
                    let mut eof = false;
                    let mut buf = [0; 8192];
                    loop {
                        match pipe.read(&mut buf) {
                            // The end of the pipe: the reader has its text.
                            Ok(0) => {
                                eof = true;
                                break;
                            }
                            Ok(n) => {
                                let kept = MAX_PIPE_BYTES.saturating_sub(bytes.len()).min(n);
                                bytes.extend_from_slice(&buf[..kept]);
                                dropped |= kept < n;
                            }
                            // A pipe without data is read again.
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                                std::thread::sleep(Duration::from_millis(1));
                            }
                            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                            Err(_) => break,
                        }
                        // The stop flag ends a reader whose pipe is still open.
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                    }
                    let mut text = String::from_utf8_lossy(&bytes).into_owned();
                    if dropped {
                        text.push_str("[truncated]");
                    }
                    let _ = tx.send(PipeRead { text, eof });
                });
                Ok((handle, rx))
            }

            let mut child = command
                .process_group(0)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;
            let stdout = child.stdout.take().expect("piped stdout");
            let stderr = child.stderr.take().expect("piped stderr");
            // The guard owns the child from here on: the drop stops it.
            let mut guard = Self {
                pgid: child.id().cast_signed(),
                child,
                reaped: false,
                reaped_at: None,
                stop_readers: Arc::new(AtomicBool::new(false)),
                readers: Vec::new(),
            };
            let stop = Arc::clone(&guard.stop_readers);
            let reader = read_pipe(stdout, Arc::clone(&stop))?;
            guard.readers.push(reader);
            let reader = read_pipe(stderr, Arc::clone(&stop))?;
            guard.readers.push(reader);
            Ok(guard)
        }

        /// Poll `try_wait` until the child exits or the deadline passes. The
        /// deadline is checked before every poll, so that a late exit is a timeout.
        fn wait(&mut self, deadline: Instant) -> Result<Option<ExitStatus>> {
            loop {
                // The child is not polled once the deadline has passed.
                if Instant::now() >= deadline {
                    return Ok(None);
                }
                match self.child.try_wait() {
                    Ok(Some(status)) => {
                        // The child is reaped: it is never signalled again, and
                        // a status that is observed after the deadline is late.
                        self.reaped = true;
                        self.reaped_at = Some(Instant::now());
                        let late = Instant::now() >= deadline;
                        return Ok(if late { None } else { Some(status) });
                    }
                    Ok(None) => {}
                    // An interrupted wait is retried, within the bound.
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(error) => return Err(error.into()),
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        /// Stop the child and its group, and reap the child, within bounds.
        fn stop(&mut self, deadline: Instant) -> Result<()> {
            // A reaped child is not polled again, and never signalled again.
            if self.reaped {
                return Ok(());
            }
            // The child is polled: a child that already left is not signalled.
            let _ = self.wait(Instant::now() + Duration::from_millis(1))?;
            if self.reaped {
                return Ok(());
            }
            // The group gets the chance to leave on its own.
            let _ = unsafe { libc::killpg(self.pgid, libc::SIGTERM) };
            let left = deadline.saturating_duration_since(Instant::now());
            let _ = self.wait(Instant::now() + left.min(Duration::from_millis(TERM_GRACE_MS)))?;
            if self.reaped {
                return Ok(());
            }
            // The `SIGTERM` was ignored: the group and the leader get the `SIGKILL`.
            let _ = unsafe { libc::killpg(self.pgid, libc::SIGKILL) };
            let _ = unsafe { libc::kill(self.child.id().cast_signed(), libc::SIGKILL) };
            // The reap is bounded: a child that does not report its exit must not block.
            let end = Instant::now() + Duration::from_secs(STOP_SECS);
            let _ = self.wait(end)?;
            ensure!(self.reaped, "child did not exit after SIGKILL");
            Ok(())
        }

        /// Collect the text of the readers, and join them, within the drain
        /// deadline. The report tells whether they reached the end of their pipes.
        fn finish_readers(&mut self, drain_deadline: Instant) -> (String, bool) {
            let mut text = String::new();
            let mut eof = true;
            for (handle, receiver) in std::mem::take(&mut self.readers) {
                let left = drain_deadline.saturating_duration_since(Instant::now());
                let Ok(part) = receiver.recv_timeout(left) else {
                    // The drain deadline has passed: the pipe is still open.
                    eof = false;
                    break;
                };
                text.push_str(&part.text);
                // The handle is joined once the thread reports itself finished.
                while !handle.is_finished() && Instant::now() < drain_deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                if !handle.is_finished() {
                    eof = false;
                    break;
                }
                let _ = handle.join();
                // A reader that stopped before the end of its pipe is reported.
                eof = eof && part.eof;
            }
            // The readers stop on every return path: no reader appends after the call.
            self.stop_readers.store(true, Ordering::Relaxed);
            (text, eof)
        }

        /// Stop a child that hung, and report the timeout of the call. The
        /// readers are stopped and joined here, so that none outlives the call.
        fn timeout_error(&mut self) -> anyhow::Error {
            let stop_by = Instant::now() + Duration::from_secs(STOP_SECS);
            let stopped = self.stop(stop_by);
            // The flag comes after the stop: the pipes of a stopped child are
            // closed, and the readers end on the end of their pipe.
            self.stop_readers.store(true, Ordering::Relaxed);
            let join_by = Instant::now() + Duration::from_secs(STOP_SECS);
            // The text of the readers of a stopped child is not reported here.
            let (_text, eof) = self.finish_readers(join_by);
            // A failure of the cleanup does not hide the timeout: it is kept.
            timeout_error_from(self.pgid, stopped, eof)
        }
    }

    #[cfg(unix)]
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let stop_by = Instant::now() + Duration::from_secs(STOP_SECS);
            if let Err(error) = self.stop(stop_by) {
                eprintln!("child cleanup: {error}");
            }
            self.stop_readers.store(true, Ordering::Relaxed);
            let join_by = Instant::now() + Duration::from_secs(STOP_SECS);
            let (_text, eof) = self.finish_readers(join_by);
            // A reader that did not reach the end of its pipe is reported.
            if !eof {
                eprintln!("child cleanup: a pipe reader did not finish");
            }
        }
    }

    /// Build the error of a timed-out child: the failures of the cleanup are
    /// kept in its chain, and the timeout is its outer context.
    #[cfg(unix)]
    fn timeout_error_from(pgid: i32, stop: Result<()>, readers_eof: bool) -> anyhow::Error {
        let mut error = stop.err();
        if !readers_eof {
            // A reader that did not reach the end of its pipe is kept too.
            let note = anyhow::anyhow!("a pipe reader did not finish");
            error = Some(match error {
                Some(error) => error.context(note),
                None => note,
            });
        }
        match error {
            // The timeout is the outer context of the failures of the cleanup.
            Some(error) => error.context(ChildTimeout(pgid)),
            None => anyhow::Error::new(ChildTimeout(pgid)),
        }
    }

    /// Run a command with a timeout, and report its status and its output.
    #[cfg(unix)]
    fn run_command(command: &mut Command, timeout: Duration) -> Result<(ExitStatus, String)> {
        let deadline = Instant::now() + timeout;
        let mut guard = ChildGuard::spawn(command)?;
        run_guard(&mut guard, deadline)
    }

    /// Run the child of a guard, and report its status and its output. The
    /// readers end on the end of their pipes: a pipe that a descendant still
    /// holds is reported after the drain deadline.
    #[cfg(unix)]
    fn run_guard(guard: &mut ChildGuard, deadline: Instant) -> Result<(ExitStatus, String)> {
        let Some(status) = guard.wait(deadline)? else {
            // The child hangs: it is stopped and reaped before the error.
            return Err(guard.timeout_error());
        };
        // The drain deadline is the reap of the leader, plus the bound.
        let reap = guard.reaped_at.unwrap_or_else(Instant::now);
        let drain_deadline = reap + Duration::from_secs(STOP_SECS);
        let (text, eof) = guard.finish_readers(drain_deadline);
        // A reader that did not reach the end of its pipe is reported.
        ensure!(eof, "a pipe reader did not finish");
        Ok((status, text))
    }

    /// Run one test of this binary again in a child process, with a deadline.
    #[cfg(unix)]
    fn run_child(name: &str, variable: &str) -> Result<(bool, String)> {
        let mut command = Command::new(std::env::current_exe()?);
        command
            .args(["--exact", name, "--nocapture"])
            .env(variable, "1");
        let (status, output) = run_command(&mut command, Duration::from_secs(CHILD_TIMEOUT_SECS))?;
        Ok((status.success(), output))
    }

    /// A child that exits at once reports its output and its status.
    #[cfg(unix)]
    #[test]
    fn a_child_that_exits_normally_reports_its_output() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "printf 'hello\\n'"]);
        let timeout = deadline.saturating_duration_since(Instant::now());
        let (status, output) = run_command(&mut command, timeout).unwrap();
        assert!(status.success());
        assert_eq!(output, "hello\n");
    }

    /// A child that ignores `SIGTERM` is stopped, and its group is gone.
    #[cfg(unix)]
    #[test]
    fn a_child_that_ignores_term_times_out_and_leaves_no_process() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut command = Command::new("/bin/sh");
        // The child ignores the `SIGTERM`, so only the `SIGKILL` stops it.
        command.args(["-c", "trap '' TERM; exec sleep 60"]);
        let error = run_command(&mut command, Duration::from_millis(300)).unwrap_err();
        assert_eq!(error.to_string(), "child timed out");
        let pgid = error
            .downcast_ref::<ChildTimeout>()
            .expect("the error of a timeout")
            .0;
        // The child and its group are gone.
        assert_group_gone(pgid, deadline);
    }

    /// The error of a timeout keeps the failures of the cleanup in its chain:
    /// the top-level message of the error stays "child timed out".
    #[cfg(unix)]
    #[test]
    fn the_error_of_a_timeout_keeps_the_failures_of_the_cleanup() {
        const STOP: &str = "child did not exit after SIGKILL";
        const READER: &str = "a pipe reader did not finish";
        let stop_error = || anyhow::anyhow!("{STOP}");
        // A stop that failed or not, and readers that ended or not.
        let cases: [(Result<()>, bool, &[&str]); 4] = [
            (Ok(()), true, &[]),
            (Err(stop_error()), true, &[STOP]),
            (Ok(()), false, &[READER]),
            (Err(stop_error()), false, &[READER, STOP]),
        ];
        for (stop, eof, notes) in cases {
            let error = timeout_error_from(7, stop, eof);
            // The timeout is the outer context of the error.
            assert_eq!(error.to_string(), "child timed out");
            assert!(error.downcast_ref::<ChildTimeout>().is_some());
            // The chain of the error keeps the failures of the cleanup.
            let chain = format!("{error:#}");
            for note in notes {
                assert!(chain.contains(note), "{chain}");
            }
        }
    }

    /// A child that writes more than a mebibyte is capped.
    #[cfg(unix)]
    #[test]
    fn a_child_that_writes_more_than_a_mebibyte_is_capped() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "head -c 2097152 /dev/zero | tr '\\0' 'a'"]);
        let timeout = deadline.saturating_duration_since(Instant::now());
        let (status, output) = run_command(&mut command, timeout).unwrap();
        assert!(status.success());
        // The output is capped, and the note names the loss.
        assert!(
            output.len() <= MAX_PIPE_BYTES + "[truncated]".len(),
            "{} bytes",
            output.len()
        );
        assert!(output.ends_with("[truncated]"), "{}", output.len());
    }

    /// The guard of a running child leaves no process behind.
    #[cfg(unix)]
    #[test]
    fn dropping_the_guard_of_a_running_child_leaves_no_process() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut command = Command::new("/bin/sleep");
        command.arg("60");
        let guard = ChildGuard::spawn(&mut command).unwrap();
        let pgid = guard.pgid;
        // The guard stops the child and reaps it.
        drop(guard);
        // The child and its group are gone.
        assert_group_gone(pgid, deadline);
    }

    /// A child that writes a line and leaves at once reports its line: the
    /// reader ends on the end of its pipe, not on the deadline of the call.
    #[cfg(unix)]
    #[test]
    fn a_reader_that_follows_a_child_that_leaves_at_once_reports_its_line() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "printf hello"]);
        let timeout = deadline.saturating_duration_since(Instant::now());
        let started = Instant::now();
        let (status, output) = run_command(&mut command, timeout).unwrap();
        let elapsed = started.elapsed();
        assert!(status.success());
        assert_eq!(output, "hello");
        // The readers do not wait for the command deadline.
        assert!(elapsed < DRAIN_CALL_BOUND, "the call took {elapsed:?}");
    }

    /// A leader that leaves while a descendant holds its pipes is reported:
    /// the readers do not reach the end of their pipes.
    #[cfg(unix)]
    #[test]
    fn a_leader_that_leaves_while_a_descendant_holds_its_pipes_is_reported() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut command = Command::new("/bin/sh");
        // The background sleep stays in the group of the leader, and holds
        // the pipes of the child after the leader has left.
        command.args(["-c", "sleep 5 & exit 0"]);
        let mut guard = ChildGuard::spawn(&mut command).unwrap();
        // The group of the child is read before the reap, so that the
        // descendant can be polled.
        let pgid = guard.pgid;
        let started = Instant::now();
        let error = run_guard(&mut guard, deadline).expect_err("the call must fail");
        let elapsed = started.elapsed();
        assert!(
            error.to_string().contains("a pipe reader did not finish"),
            "{error}"
        );
        // The readers do not wait for the command deadline.
        assert!(elapsed < DRAIN_CALL_BOUND, "the call took {elapsed:?}");
        // The descendant leaves on its own: the group is polled, and this
        // test sends no signal.
        assert_group_gone(pgid, deadline);
    }

    /// An adapter for the pinned Qwen3.5 text subset.
    fn text_subset_adapter() -> Result<Qwen35Adapter> {
        let config: crate::models::qwen3::Qwen3Config =
            serde_json::from_value(serde_json::json!({
                "model_type":"qwen3", "vocab_size":128, "hidden_size":64,
                "intermediate_size":128, "num_hidden_layers":1, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":16, "max_position_embeddings":128,
                "tie_word_embeddings":true, "rms_norm_eps":0.000_001, "hidden_act":"silu"
            }))?;
        let mut spec = config.to_spec()?;
        spec.architecture = "qwen35".into();
        Ok(Qwen35Adapter {
            spec,
            chat_template: Some(Qwen35ChatTemplate::OfficialTextSubset),
            jinja: None,
        })
    }
    #[cfg(unix)]
    #[test]
    fn official_text_subset_logs_pinned_path() -> Result<()> {
        const CHILD: &str = "MACLLM_QWEN35_PINNED_LOG_TEST";
        if std::env::var_os(CHILD).is_some() {
            let mut adapter = text_subset_adapter()?;
            adapter.attach_prompt_tokens(GgufPromptTokens {
                chat_template: Some("{{ messages }}"),
                ..Default::default()
            })?;
            // The chat path is pinned, and the generic renderer stays
            // available for the agent path.
            assert!(adapter.jinja.is_some());
            assert!(
                adapter
                    .format_chat_messages(&[ChatMessage::user("hello")])?
                    .contains("hello")
            );
        } else {
            let (success, output) = run_child(
                "models::qwen35::tests::official_text_subset_logs_pinned_path",
                CHILD,
            )?;
            assert!(success, "{output}");
            assert!(
                output.contains("chat template: family=qwen35 path=pinned"),
                "{output}"
            );
        }
        Ok(())
    }

    #[test]
    fn official_text_subset_keeps_generic_tool_calling() -> Result<()> {
        use crate::agent::{AgentDirective, AgentMessage, AgentPrompt, ToolDefinition};
        let source = "{# <tool_call> \"name\": \"arguments\": #}{{ tools | tojson }}{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}";
        let mut adapter = text_subset_adapter()?;
        adapter.attach_prompt_tokens(GgufPromptTokens {
            chat_template: Some(source),
            ..Default::default()
        })?;
        // The pinned chat template has no agent profile, so the generic
        // renderer and its detected format serve the tool calls.
        assert_eq!(
            adapter.tool_call_format(),
            Some(macllm_chat::tool_format::ToolCallFormat::Hermes)
        );
        let prompt = AgentPrompt {
            messages: vec![AgentMessage::User {
                content: "go".into(),
            }],
            tools: vec![ToolDefinition {
                name: "weather".into(),
                description: Some("forecast".into()),
                parameters: serde_json::json!({"type":"object"}),
                strict: false,
                declaration: None,
            }],
            directive: AgentDirective::Auto,
        };
        let rendered = adapter.format_agent_prompt(&prompt)?;
        assert!(rendered.contains("\"type\": \"function\""), "{rendered}");
        assert!(rendered.contains("\"name\": \"weather\""), "{rendered}");
        Ok(())
    }

    #[test]
    fn renders_official_plain_text_default_non_thinking_subset() -> Result<()> {
        let prompt = format_qwen35_chat_messages(&[
            ChatMessage::system("  Be concise.  "),
            ChatMessage::user(" First? "),
            ChatMessage::assistant("<think>old reasoning</think>\n\nFirst. "),
            ChatMessage::user("Second?"),
        ])?;
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nFirst?<|im_end|>\n<|im_start|>assistant\nFirst.<|im_end|>\n<|im_start|>user\nSecond?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            format_qwen35_chat_messages(&[ChatMessage::user("What is the capital of France?")])?,
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        Ok(())
    }

    #[test]
    fn recognizes_exact_new_identities() {
        for (name, expected) in [
            ("official", Qwen35ChatTemplate::Official),
            ("unsloth", Qwen35ChatTemplate::Unsloth),
        ] {
            let path = format!(
                "{}/tests/fixtures/chat_templates/templates/qwen35-{name}.jinja",
                env!("CARGO_MANIFEST_DIR")
            );
            let mut bytes = std::fs::read(path).unwrap();
            assert_eq!(
                classify_chat_template(Some(std::str::from_utf8(&bytes).unwrap())),
                Some(expected)
            );
            bytes[0] ^= 1;
            assert_eq!(
                classify_chat_template(Some(std::str::from_utf8(&bytes).unwrap())),
                None
            );
        }
    }

    #[test]
    fn unknown_template_stays_raw_only() {
        assert_eq!(classify_chat_template(None), None);
        assert_eq!(classify_chat_template(Some("unverified")), None);
    }

    /// Poll until the process group of the child is gone, or the deadline.
    #[cfg(unix)]
    fn group_gone(pgid: i32, deadline: Instant) -> bool {
        loop {
            let gone = unsafe { libc::kill(-pgid, 0) } == -1
                && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
            if gone {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The process group of the child is gone, within the deadline.
    #[cfg(unix)]
    fn assert_group_gone(pgid: i32, deadline: Instant) {
        assert!(
            group_gone(pgid, deadline),
            "process group {pgid} still exists"
        );
    }
}
