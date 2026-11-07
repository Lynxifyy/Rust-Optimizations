//! Read a per-build offset map and decide, fail-closed, whether it is usable.
//!
//! The `tools/extract_offsets.py` pipeline emits one JSON file per game build
//! (`offsets_26_60.json`). This module parses that shape and answers the only
//! question the mod asks: **can I trust these offsets for the build I am in?**
//!
//! Fail-closed is the whole design. An offset map is only usable when the file's
//! `so_sha256` matches the running `libminecraftpe.so` and every offset the
//! caller needs is present. A mismatch (a game update, a wrong file) or a missing
//! offset is a `None`, and the caller keeps the `FlatWorldGenerator` fallback.
//! There is no "probably fine" path - a wrong offset is a fault on the game
//! thread, which is worse than not having the feature.
//!
//! The parser is deliberately hand-written and dependency-free so it can be
//! host-tested; the JSON is a flat, machine-generated document, not arbitrary
//! input.

use std::collections::BTreeMap;

/// One resolved RTTI vtable slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RttiSlot {
    pub type_name: String,
    pub slot: u32,
    pub function_vaddr: u64,
}

/// A parsed offset map for one build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OffsetMap {
    pub version: String,
    pub so_sha256: String,
    pub so_bytes: u64,
    /// Keyed by `"<type_name>#<slot>"`.
    pub rtti_slots: BTreeMap<String, RttiSlot>,
    /// The `EntitySeamOffsets` fields, present only when the disassembly pass has
    /// recovered them. Empty means "not recovered", which is the honest default.
    pub entity_seam: BTreeMap<String, u64>,
    /// Signature names that resolved to exactly one `.text` location.
    pub unique_signatures: Vec<String>,
}

/// Why a map was rejected. Reported, never swallowed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OffsetMapError {
    Parse(String),
    /// The map is for a different `libminecraftpe.so` than the running one.
    BuildMismatch {
        expected: String,
        actual: String,
    },
}

impl std::fmt::Display for OffsetMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OffsetMapError::Parse(message) => write!(f, "offset map parse error: {message}"),
            OffsetMapError::BuildMismatch { expected, actual } => write!(
                f,
                "offset map build mismatch: map is for {expected}, running {actual}"
            ),
        }
    }
}

// --- A very small JSON reader ------------------------------------------------
//
// The document is machine-generated and flat, so a full JSON parser would be
// overkill. This extracts the string/number values by key at a fixed nesting and
// ignores everything else. It is intentionally strict about the keys it needs and
// lenient about the rest, so an added field in a future report does not break it.

fn json_string_after(haystack: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let at = haystack.find(&needle)?;
    let after = &haystack[at + needle.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn json_number_after(haystack: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let at = haystack.find(&needle)?;
    let after = &haystack[at + needle.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Extracts the objects of a named array, as raw substrings.
fn json_array_objects(haystack: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\"");
    let Some(at) = haystack.find(&needle) else {
        return Vec::new();
    };
    let after = &haystack[at + needle.len()..];
    let Some(bracket) = after.find('[') else {
        return Vec::new();
    };
    let mut rest = &after[bracket + 1..];
    let mut out = Vec::new();
    loop {
        let trimmed = rest.trim_start();
        if trimmed.starts_with(']') || trimmed.is_empty() {
            break;
        }
        if !trimmed.starts_with('{') {
            break;
        }
        let mut depth = 0i32;
        let mut end = None;
        for (index, ch) in trimmed.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(index + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else { break };
        out.push(trimmed[..end].to_string());
        // Skip the separating comma so the next element is reached.
        rest = trimmed[end..].trim_start();
        rest = rest.strip_prefix(',').unwrap_or(rest);
    }
    out
}

impl OffsetMap {
    /// Parses a report. Does not check the build - call [`OffsetMap::verify_build`].
    pub fn parse(json: &str) -> Result<Self, OffsetMapError> {
        let version = json_string_after(json, "version")
            .ok_or_else(|| OffsetMapError::Parse("missing version".into()))?;
        let so_sha256 = json_string_after(json, "so_sha256")
            .ok_or_else(|| OffsetMapError::Parse("missing so_sha256".into()))?;
        let so_bytes = json_number_after(json, "so_bytes").unwrap_or(0);

        let mut rtti_slots = BTreeMap::new();
        for object in json_array_objects(json, "rtti_slots") {
            let (Some(type_name), Some(slot), Some(vaddr)) = (
                json_string_after(&object, "type_name"),
                json_number_after(&object, "slot"),
                json_number_after(&object, "function_vaddr"),
            ) else {
                continue;
            };
            rtti_slots.insert(
                format!("{type_name}#{slot}"),
                RttiSlot {
                    type_name,
                    slot: slot as u32,
                    function_vaddr: vaddr,
                },
            );
        }

        let mut entity_seam = BTreeMap::new();
        for object in json_array_objects(json, "fields") {
            let (Some(name), Some(offset)) = (
                json_string_after(&object, "name"),
                json_number_after(&object, "offset"),
            ) else {
                continue;
            };
            // A `null` offset (unrecovered) has no number, so it is simply absent.
            entity_seam.insert(name, offset);
        }

        let mut unique_signatures = Vec::new();
        for object in json_array_objects(json, "signatures") {
            if let Some(name) = json_string_after(&object, "name") {
                if json_string_after(&object, "status").as_deref() == Some("unique") {
                    unique_signatures.push(name);
                }
            }
        }

        Ok(Self {
            version,
            so_sha256,
            so_bytes,
            rtti_slots,
            entity_seam,
            unique_signatures,
        })
    }

    /// True when the map describes the given build id.
    pub fn verify_build(&self, running_sha256: &str) -> Result<(), OffsetMapError> {
        if self.so_sha256.eq_ignore_ascii_case(running_sha256) {
            Ok(())
        } else {
            Err(OffsetMapError::BuildMismatch {
                expected: self.so_sha256.clone(),
                actual: running_sha256.to_string(),
            })
        }
    }

    /// The function address of a resolved slot, if the map has it.
    pub fn slot(&self, type_name: &str, slot: u32) -> Option<u64> {
        self.rtti_slots
            .get(&format!("{type_name}#{slot}"))
            .map(|entry| entry.function_vaddr)
    }

    /// True when the entity seam is fully recovered (all four offsets present and
    /// non-zero). Until then the seam stays null and the fallback is used.
    pub fn entity_seam_complete(&self) -> bool {
        const REQUIRED: [&str; 4] = [
            "levelEntityListOffset",
            "entityRuntimeIdOffset",
            "entityPositionOffset",
            "entityRotationOffset",
        ];
        REQUIRED
            .iter()
            .all(|name| self.entity_seam.get(*name).is_some_and(|v| *v != 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> String {
        r#"{
  "version": "26.60",
  "so_bytes": 342231392,
  "so_sha256": "b6ffe00817117b03a518b4c9254dcbbb821ad53f276d179c67021226ca221a80",
  "abi": "arm64-v8a",
  "rtti_slots": [
    {"type_name": "14ClientInstance", "slot": 31, "function_vaddr": 163051024, "vtable_vaddr": 327745128, "resolved_by": "slot-index"},
    {"type_name": "18LivePlayerRenderer", "slot": 17, "function_vaddr": 161642012, "vtable_vaddr": 327680744, "resolved_by": "slot-index"}
  ],
  "rtti_slots_by_needle": [],
  "signatures": [
    {"name": "some_unique_sig", "status": "unique", "match_count": 1},
    {"name": "prologue", "status": "ambiguous", "match_count": 43678}
  ],
  "entity_seam_offsets": {
    "resolved": false,
    "reason": "requires disassembly",
    "fields": [
      {"name": "levelEntityListOffset", "meaning": "x", "offset": null},
      {"name": "entityRuntimeIdOffset", "meaning": "x", "offset": null}
    ]
  }
}"#
        .to_string()
    }

    #[test]
    fn parses_the_core_fields() {
        let map = OffsetMap::parse(&sample()).unwrap();
        assert_eq!(map.version, "26.60");
        assert_eq!(map.so_bytes, 342231392);
        assert!(map.so_sha256.starts_with("b6ffe008"));
    }

    #[test]
    fn parses_rtti_slots_keyed_by_type_and_slot() {
        let map = OffsetMap::parse(&sample()).unwrap();
        assert_eq!(map.rtti_slots.len(), 2);
        assert_eq!(map.slot("14ClientInstance", 31), Some(163051024));
        assert_eq!(map.slot("18LivePlayerRenderer", 17), Some(161642012));
        assert_eq!(map.slot("14ClientInstance", 999), None);
    }

    #[test]
    fn records_only_unique_signatures() {
        let map = OffsetMap::parse(&sample()).unwrap();
        assert_eq!(map.unique_signatures, vec!["some_unique_sig".to_string()]);
    }

    #[test]
    fn unrecovered_entity_seam_is_not_complete() {
        let map = OffsetMap::parse(&sample()).unwrap();
        assert!(!map.entity_seam_complete());
        // The null offsets are simply absent, not zero-filled.
        assert!(map.entity_seam.is_empty());
    }

    #[test]
    fn a_fully_recovered_entity_seam_is_complete() {
        let json = r#"{"version":"x","so_sha256":"abc","entity_seam_offsets":{"fields":[
            {"name":"levelEntityListOffset","offset":1234},
            {"name":"entityRuntimeIdOffset","offset":8},
            {"name":"entityPositionOffset","offset":560},
            {"name":"entityRotationOffset","offset":568}
        ]}}"#;
        let map = OffsetMap::parse(json).unwrap();
        assert!(map.entity_seam_complete());
    }

    #[test]
    fn a_zero_offset_is_treated_as_unrecovered() {
        let json = r#"{"version":"x","so_sha256":"abc","entity_seam_offsets":{"fields":[
            {"name":"levelEntityListOffset","offset":0},
            {"name":"entityRuntimeIdOffset","offset":8},
            {"name":"entityPositionOffset","offset":560},
            {"name":"entityRotationOffset","offset":568}
        ]}}"#;
        let map = OffsetMap::parse(json).unwrap();
        assert!(
            !map.entity_seam_complete(),
            "a 0 offset is the 'not recovered' sentinel"
        );
    }

    #[test]
    fn build_verification_accepts_a_match_and_rejects_a_mismatch() {
        let map = OffsetMap::parse(&sample()).unwrap();
        assert!(map
            .verify_build("B6FFE00817117B03A518B4C9254DCBBB821AD53F276D179C67021226CA221A80")
            .is_ok());
        let err = map.verify_build("deadbeef").unwrap_err();
        assert!(matches!(err, OffsetMapError::BuildMismatch { .. }));
        assert!(err.to_string().contains("mismatch"));
    }

    #[test]
    fn missing_required_fields_are_a_parse_error() {
        assert!(matches!(
            OffsetMap::parse(r#"{"so_sha256":"x"}"#),
            Err(OffsetMapError::Parse(_))
        ));
        assert!(matches!(
            OffsetMap::parse(r#"{"version":"x"}"#),
            Err(OffsetMapError::Parse(_))
        ));
    }

    #[test]
    fn extra_unknown_fields_are_ignored() {
        let json =
            r#"{"version":"26.60","so_sha256":"abc","future_field":[1,2,3],"nested":{"a":1}}"#;
        let map = OffsetMap::parse(json).unwrap();
        assert_eq!(map.version, "26.60");
        assert!(map.rtti_slots.is_empty());
    }

    #[test]
    fn malformed_entries_are_skipped_not_fatal() {
        // An rtti entry missing function_vaddr must not take the whole parse down.
        let json = r#"{"version":"x","so_sha256":"abc","rtti_slots":[
            {"type_name":"A","slot":1},
            {"type_name":"B","slot":2,"function_vaddr":4096}
        ]}"#;
        let map = OffsetMap::parse(json).unwrap();
        assert_eq!(map.rtti_slots.len(), 1);
        assert_eq!(map.slot("B", 2), Some(4096));
    }
}
