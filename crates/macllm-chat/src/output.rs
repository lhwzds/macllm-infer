use std::{error::Error, fmt, mem};

/// `OpenAI` Chat Completions accepts at most four caller-provided stop strings.
pub const MAX_STOP_SEQUENCES: usize = 4;

/// Invalid configuration supplied while constructing a [`StopFilter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopFilterConfigError {
    TooManyStops { count: usize },
    EmptyStop { index: usize },
}

impl fmt::Display for StopFilterConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyStops { count } => write!(
                formatter,
                "stop sequences contain {count} entries; at most {MAX_STOP_SEQUENCES} are supported"
            ),
            Self::EmptyStop { index } => {
                write!(
                    formatter,
                    "stop sequence at index {index} must not be empty"
                )
            }
        }
    }
}

impl Error for StopFilterConfigError {}

/// One incremental filtering result.
///
/// `text` is safe to expose immediately: no suffix in it can become part of a
/// stop match after a later fragment arrives. When `matched` is true, the
/// request has reached a configured stop and must not feed more model output to
/// the client.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StopFilterOutput {
    pub text: String,
    pub matched: bool,
}

impl StopFilterOutput {
    fn emitted(text: String) -> Self {
        Self {
            text,
            matched: false,
        }
    }

    fn matched(text: String) -> Self {
        Self {
            text,
            matched: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopFilterState {
    Open,
    Matched(usize),
    Finished,
}

/// Request-local incremental filter for decoded UTF-8 model output.
///
/// The filter withholds only the longest suffix that is also a prefix of a
/// configured stop. Everything before that suffix is safe to emit. A complete
/// stop and all text decoded after it are suppressed. Call [`Self::finish`] on
/// normal EOS or length termination to release a withheld partial prefix.
#[derive(Debug, Clone)]
pub struct StopFilter {
    stops: Vec<String>,
    maximum_stop_bytes: usize,
    pending: String,
    state: StopFilterState,
}

impl StopFilter {
    /// Validate and construct a filter for zero to four non-empty stop strings.
    pub fn new(stops: Vec<String>) -> Result<Self, StopFilterConfigError> {
        if stops.len() > MAX_STOP_SEQUENCES {
            return Err(StopFilterConfigError::TooManyStops { count: stops.len() });
        }
        if let Some(index) = stops.iter().position(String::is_empty) {
            return Err(StopFilterConfigError::EmptyStop { index });
        }

        let maximum_stop_bytes = stops.iter().map(String::len).max().unwrap_or(0);
        Ok(Self {
            stops,
            maximum_stop_bytes,
            pending: String::new(),
            state: StopFilterState::Open,
        })
    }

    /// Consume one decoded UTF-8 fragment and return only text safe to emit.
    ///
    /// The first stop to complete in stream order wins. If several stops
    /// complete at the same byte position, the one beginning earliest wins;
    /// duplicate matches are resolved by their configuration order. This makes
    /// behavior independent of how the same decoded text is split into input
    /// fragments.
    pub fn push(&mut self, fragment: &str) -> StopFilterOutput {
        match self.state {
            StopFilterState::Matched(_) => return StopFilterOutput::matched(String::new()),
            StopFilterState::Finished => return StopFilterOutput::default(),
            StopFilterState::Open => {}
        }

        self.pending.push_str(fragment);
        if let Some((start, stop_index)) = self.earliest_completed_stop() {
            self.pending.truncate(start);
            self.state = StopFilterState::Matched(stop_index);
            return StopFilterOutput::matched(mem::take(&mut self.pending));
        }

        let held_start = self.longest_possible_stop_prefix_start();
        let held = self.pending.split_off(held_start);
        let safe = mem::replace(&mut self.pending, held);
        StopFilterOutput::emitted(safe)
    }

    /// Complete normal EOS/length handling and release any partial stop prefix.
    ///
    /// A matched filter never releases the matched stop. The method is
    /// idempotent; subsequent calls return an empty result.
    pub fn finish(&mut self) -> StopFilterOutput {
        match self.state {
            StopFilterState::Open => {
                self.state = StopFilterState::Finished;
                StopFilterOutput::emitted(mem::take(&mut self.pending))
            }
            StopFilterState::Matched(_) => StopFilterOutput::matched(String::new()),
            StopFilterState::Finished => StopFilterOutput::default(),
        }
    }

    /// Whether either a configured stop matched or normal output was finished.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        !matches!(self.state, StopFilterState::Open)
    }

    /// Whether output terminated because a configured stop matched.
    #[must_use]
    pub const fn is_matched(&self) -> bool {
        matches!(self.state, StopFilterState::Matched(_))
    }

    /// The configured stop that terminated this request, if any.
    #[must_use]
    pub fn matched_stop(&self) -> Option<&str> {
        match self.state {
            StopFilterState::Matched(index) => Some(&self.stops[index]),
            StopFilterState::Open | StopFilterState::Finished => None,
        }
    }

    /// Find the stop that would have completed first if `pending` had arrived
    /// one Unicode scalar at a time. `str::find` yields UTF-8 boundaries, so the
    /// returned truncation point always keeps `pending` valid UTF-8.
    fn earliest_completed_stop(&self) -> Option<(usize, usize)> {
        self.stops
            .iter()
            .enumerate()
            .filter_map(|(index, stop)| {
                self.pending.find(stop).map(|start| {
                    let end = start + stop.len();
                    ((end, start, index), (start, index))
                })
            })
            .min_by_key(|(order, _)| *order)
            .map(|(_, matched)| matched)
    }

    /// Locate the longest suffix that could still become a stop after future
    /// fragments. Only the last `maximum_stop_bytes` can participate, bounding
    /// the scan even when callers pass a large decoded fragment.
    fn longest_possible_stop_prefix_start(&self) -> usize {
        if self.stops.is_empty() || self.pending.is_empty() {
            return self.pending.len();
        }

        let first_possible_byte = self.pending.len().saturating_sub(self.maximum_stop_bytes);
        self.pending
            .char_indices()
            .filter(|(start, _)| *start >= first_possible_byte)
            .find_map(|(start, _)| {
                let suffix = &self.pending[start..];
                self.stops
                    .iter()
                    .any(|stop| stop.starts_with(suffix))
                    .then_some(start)
            })
            .unwrap_or(self.pending.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn validates_count_and_non_empty_stops() {
        assert!(StopFilter::new(Vec::new()).is_ok());
        assert!(StopFilter::new(strings(&["a", "b", "c", "d"])).is_ok());
        assert_eq!(
            StopFilter::new(strings(&["a", "b", "c", "d", "e"])).unwrap_err(),
            StopFilterConfigError::TooManyStops { count: 5 }
        );
        assert_eq!(
            StopFilter::new(strings(&["valid", "", "later"])).unwrap_err(),
            StopFilterConfigError::EmptyStop { index: 1 }
        );
    }

    #[test]
    fn no_stops_emit_every_fragment_immediately() {
        let mut filter = StopFilter::new(Vec::new()).unwrap();
        assert_eq!(
            filter.push("hello"),
            StopFilterOutput::emitted("hello".to_owned())
        );
        assert_eq!(
            filter.push(" world"),
            StopFilterOutput::emitted(" world".to_owned())
        );
        assert_eq!(filter.finish(), StopFilterOutput::default());
        assert!(filter.is_finished());
        assert!(!filter.is_matched());
    }

    #[test]
    fn complete_stop_inside_one_fragment_suppresses_stop_and_suffix() {
        let mut filter = StopFilter::new(strings(&["STOP"])).unwrap();
        assert_eq!(
            filter.push("before STOP after"),
            StopFilterOutput::matched("before ".to_owned())
        );
        assert!(filter.is_finished());
        assert!(filter.is_matched());
        assert_eq!(filter.matched_stop(), Some("STOP"));
        assert_eq!(filter.finish(), StopFilterOutput::matched(String::new()));
    }

    #[test]
    fn stop_across_fragments_never_leaks_a_possible_prefix() {
        let mut filter = StopFilter::new(strings(&["STOP"])).unwrap();
        assert_eq!(
            filter.push("before ST"),
            StopFilterOutput::emitted("before ".to_owned())
        );
        assert_eq!(filter.push("O"), StopFilterOutput::default());
        assert_eq!(
            filter.push("P after"),
            StopFilterOutput::matched(String::new())
        );
        assert_eq!(filter.matched_stop(), Some("STOP"));
    }

    #[test]
    fn normal_finish_flushes_an_incomplete_stop_prefix_once() {
        let mut filter = StopFilter::new(strings(&["STOP"])).unwrap();
        assert_eq!(
            filter.push("before ST"),
            StopFilterOutput::emitted("before ".to_owned())
        );
        assert_eq!(filter.finish(), StopFilterOutput::emitted("ST".to_owned()));
        assert_eq!(filter.finish(), StopFilterOutput::default());
        assert_eq!(filter.push("ignored"), StopFilterOutput::default());
        assert_eq!(filter.matched_stop(), None);
    }

    #[test]
    fn partial_match_can_fall_back_to_safe_text() {
        let mut filter = StopFilter::new(strings(&["abab"])).unwrap();
        assert_eq!(
            filter.push("prefix aba"),
            StopFilterOutput::emitted("prefix ".to_owned())
        );
        assert_eq!(
            filter.push("x"),
            StopFilterOutput::emitted("abax".to_owned())
        );
        assert_eq!(filter.finish(), StopFilterOutput::default());
    }

    #[test]
    fn retains_only_the_longest_suffix_that_can_still_match() {
        let mut filter = StopFilter::new(strings(&["abcde", "cdef"])).unwrap();
        assert_eq!(
            filter.push("safe abc"),
            StopFilterOutput::emitted("safe ".to_owned())
        );
        assert_eq!(
            filter.push("x cde"),
            StopFilterOutput::emitted("abcx ".to_owned())
        );
        assert_eq!(filter.finish(), StopFilterOutput::emitted("cde".to_owned()));
    }

    #[test]
    fn unicode_stop_matches_across_utf8_fragments_without_splitting_text() {
        let mut filter = StopFilter::new(strings(&["终止"])).unwrap();
        assert_eq!(
            filter.push("答案终"),
            StopFilterOutput::emitted("答案".to_owned())
        );
        assert_eq!(
            filter.push("止不会显示"),
            StopFilterOutput::matched(String::new())
        );
        assert_eq!(filter.matched_stop(), Some("终止"));
    }

    #[test]
    fn unicode_partial_prefix_is_flushed_on_normal_finish() {
        let mut filter = StopFilter::new(strings(&["🛑结束"])).unwrap();
        assert_eq!(
            filter.push("保留🛑"),
            StopFilterOutput::emitted("保留".to_owned())
        );
        assert_eq!(filter.finish(), StopFilterOutput::emitted("🛑".to_owned()));
    }

    #[test]
    fn shorter_shared_prefix_stop_completes_without_waiting_for_longer_one() {
        let mut filter = StopFilter::new(strings(&["END", "ENDING"])).unwrap();
        assert_eq!(
            filter.push("value END"),
            StopFilterOutput::matched("value ".to_owned())
        );
        assert_eq!(filter.matched_stop(), Some("END"));
    }

    #[test]
    fn earliest_completion_wins_over_a_longer_earlier_start() {
        let mut filter = StopFilter::new(strings(&["abcdef", "bc"])).unwrap();
        assert_eq!(
            filter.push("abcdef trailing"),
            StopFilterOutput::matched("a".to_owned())
        );
        assert_eq!(filter.matched_stop(), Some("bc"));
    }

    #[test]
    fn same_completion_prefers_the_match_with_the_earliest_start() {
        let mut filter = StopFilter::new(strings(&["bc", "abc"])).unwrap();
        assert_eq!(
            filter.push("abc trailing"),
            StopFilterOutput::matched(String::new())
        );
        assert_eq!(filter.matched_stop(), Some("abc"));
    }

    #[test]
    fn duplicate_stops_are_resolved_by_configuration_order() {
        let mut filter = StopFilter::new(strings(&["same", "same"])).unwrap();
        assert_eq!(
            filter.push("prefix same"),
            StopFilterOutput::matched("prefix ".to_owned())
        );
        assert_eq!(filter.state, StopFilterState::Matched(0));
    }

    #[test]
    fn earliest_repeated_occurrence_is_used() {
        let mut filter = StopFilter::new(strings(&["aba"])).unwrap();
        assert_eq!(
            filter.push("xxababa"),
            StopFilterOutput::matched("xx".to_owned())
        );
    }

    #[test]
    fn empty_fragments_do_not_disturb_a_held_prefix() {
        let mut filter = StopFilter::new(strings(&["done"])).unwrap();
        assert_eq!(filter.push("do"), StopFilterOutput::default());
        assert_eq!(filter.push(""), StopFilterOutput::default());
        assert_eq!(
            filter.push("x"),
            StopFilterOutput::emitted("dox".to_owned())
        );
    }

    #[test]
    fn pushes_after_a_match_cannot_emit_more_text() {
        let mut filter = StopFilter::new(strings(&["!"])).unwrap();
        assert_eq!(
            filter.push("hello!ignored"),
            StopFilterOutput::matched("hello".to_owned())
        );
        assert_eq!(
            filter.push("more"),
            StopFilterOutput::matched(String::new())
        );
        assert_eq!(filter.finish(), StopFilterOutput::matched(String::new()));
    }

    #[test]
    fn behavior_is_independent_of_fragment_boundaries() {
        fn collect(parts: &[&str]) -> (String, Option<String>) {
            let mut filter = StopFilter::new(strings(&["abcdef", "bc", "终止"])).unwrap();
            let mut emitted = String::new();
            for part in parts {
                let output = filter.push(part);
                emitted.push_str(&output.text);
                if output.matched {
                    break;
                }
            }
            if !filter.is_matched() {
                emitted.push_str(&filter.finish().text);
            }
            (emitted, filter.matched_stop().map(str::to_owned))
        }

        let expected = collect(&["prefix abcdef suffix"]);
        assert_eq!(expected, ("prefix a".to_owned(), Some("bc".to_owned())));
        assert_eq!(collect(&["prefix a", "b", "c", "def suffix"]), expected);
        assert_eq!(collect(&["prefix ab", "cdef", " suffix"]), expected);
    }
}
