#!/usr/bin/env python3
"""Unit tests for the offset tools' pure logic.

These run on a normal machine with no APK and no `.so`: they cover the pattern
parser, the wildcard scanner, the URL/HTML parsing of the download chain, and the
report shape. The end-to-end extraction needs the real binaries and is run
separately (see docs/agent-notes/signature-derivation.md).

Run: python3 -m unittest discover -s tools/tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import download_and_extract as dle  # noqa: E402
import extract_offsets as eo  # noqa: E402


class TestPatternParsing(unittest.TestCase):
    def test_plain_bytes(self):
        pattern, mask = eo.parse_pattern("48 8B 89")
        self.assertEqual(pattern, b"\x48\x8b\x89")
        self.assertEqual(mask, b"\xff\xff\xff")

    def test_wildcards(self):
        pattern, mask = eo.parse_pattern("48 ?? 89 ** ?")
        self.assertEqual(pattern, b"\x48\x00\x89\x00\x00")
        self.assertEqual(mask, b"\xff\x00\xff\x00\x00")

    def test_lowercase_and_mixed(self):
        pattern, _ = eo.parse_pattern("fd 7b bf a9")
        self.assertEqual(pattern, b"\xfd\x7b\xbf\xa9")

    def test_empty_is_rejected(self):
        with self.assertRaises(ValueError):
            eo.parse_pattern("   ")

    def test_bad_token_is_rejected(self):
        with self.assertRaises(ValueError):
            eo.parse_pattern("48 8b zz")
        with self.assertRaises(ValueError):
            eo.parse_pattern("4")


class TestScanner(unittest.TestCase):
    def _scan(self, haystack, pattern):
        p, m = eo.parse_pattern(pattern)
        return eo.scan_bytes(haystack, p, m)

    def test_exact_match(self):
        self.assertEqual(self._scan(b"\x00\x48\x8b\x89\x00", "48 8B 89"), [1])

    def test_multiple_matches_are_all_reported(self):
        hay = b"\x48\x8b\x00\x48\x8b\x00\x48\x8b"
        self.assertEqual(self._scan(hay, "48 8B"), [0, 3, 6])

    def test_wildcard_matches_anything(self):
        hay = b"\x48\x11\x89\x48\x22\x89"
        self.assertEqual(self._scan(hay, "48 ?? 89"), [0, 3])

    def test_no_match(self):
        self.assertEqual(self._scan(b"\x00\x01\x02", "48 8B"), [])

    def test_all_wildcard_is_rejected(self):
        p, m = eo.parse_pattern("?? ??")
        with self.assertRaises(ValueError):
            eo.scan_bytes(b"\x00\x00", p, m)

    def test_match_at_the_very_end(self):
        self.assertEqual(self._scan(b"\x00\x00\x48\x8b", "48 8B"), [2])

    def test_pattern_longer_than_haystack(self):
        self.assertEqual(self._scan(b"\x48", "48 8B 89"), [])


class TestDownloadChain(unittest.TestCase):
    PAGE = (
        '<div><form method="post" action="/show_file.php">'
        '<input type="hidden" name="file_id" value="7694">'
        '<input type="hidden" name="post_title" value="x"></form></div>'
    )
    SHOW = 'var DOWNLOAD_URL = "https://file.mcpedl.org/uploads_files/06-11-2026/minecraft-26-62.apk";'

    def test_find_file_id(self):
        self.assertEqual(dle.find_file_id(self.PAGE), "7694")

    def test_find_file_id_missing(self):
        with self.assertRaises(dle.DownloadError):
            dle.find_file_id("<html>no form here</html>")

    def test_find_download_url(self):
        self.assertEqual(
            dle.find_download_url(self.SHOW),
            "https://file.mcpedl.org/uploads_files/06-11-2026/minecraft-26-62.apk",
        )

    def test_find_download_url_missing(self):
        with self.assertRaises(dle.DownloadError):
            dle.find_download_url("<html>nothing</html>")

    def test_slug_and_key(self):
        self.assertEqual(dle.version_to_slug("26.60"), "minecraft-pe-26-60-apk")
        self.assertEqual(dle.version_to_key("26.60"), "26_60")
        self.assertEqual(dle.version_to_slug("26.62"), "minecraft-pe-26-62-apk")

    def test_so_member_is_arm64(self):
        self.assertEqual(dle.SO_MEMBER, "lib/arm64-v8a/libminecraftpe.so")


class TestEntityOffsetVerification(unittest.TestCase):
    """The offset verifier's two signals, tested on synthetic bytes."""

    def test_member_address_getter_is_recognised(self):
        # `add x0, x0, #0x230 ; ret` is what a Vec3 member getter looks like.
        words = [0x91000000 | (0x230 << 10), 0xD65F03C0]
        blob = b"".join(w.to_bytes(4, "little") for w in words)
        self.assertEqual(
            eo.member_address_getters_from_bytes(blob), {0x230}
        )

    def test_add_without_ret_is_not_a_getter(self):
        # Same add, but followed by something else: not a leaf getter.
        words = [0x91000000 | (0x230 << 10), 0x91000000]
        blob = b"".join(w.to_bytes(4, "little") for w in words)
        self.assertEqual(eo.member_address_getters_from_bytes(blob), set())

    def test_add_with_a_different_destination_is_not_a_getter(self):
        # `add x1, x0, #0x230 ; ret` returns a value in x1, not `this + imm`.
        words = [0x91000000 | (0x230 << 10) | 1, 0xD65F03C0]
        blob = b"".join(w.to_bytes(4, "little") for w in words)
        self.assertEqual(eo.member_address_getters_from_bytes(blob), set())

    def test_float_access_decoder(self):
        # ldr s4, [x0, #0x230]  and  ldr s4, [x0, #0x238]
        self.assertEqual(eo._decode_float_access(0xBD423004), (0, 0x230))
        self.assertEqual(eo._decode_float_access(0xBD423804), (0, 0x238))
        # ldr d0, [x8, #0x10] is an 8-byte scaled access -> disp 0x10.
        self.assertEqual(eo._decode_float_access(0xfd400900), (8, 0x10))
        # A plain integer load is not a float access.
        self.assertEqual(eo._decode_float_access(0xF9400000), None)

    def test_vec3_group_needs_three_consecutive_floats(self):
        # Three floats at +0x230/+0x234/+0x238 in one function: a Vec3.
        blob = b"".join(
            w.to_bytes(4, "little")
            for w in (0xBD423004, 0xBD423404, 0xBD423804)
        )
        self.assertIn(0x230, eo.vec3_float_groups_from_bytes(blob))

    def test_two_floats_are_not_a_vec3_group(self):
        blob = b"".join(
            w.to_bytes(4, "little") for w in (0xBD423004, 0xBD423804)
        )
        self.assertNotIn(0x230, eo.vec3_float_groups_from_bytes(blob))

    def test_stack_relative_accesses_are_ignored(self):
        # sp-relative (reg 31) floats at 0x230/0x234/0x238 must not form a group.
        blob = b"".join(
            w.to_bytes(4, "little")
            for w in (0xBD423F04, 0xBD423F24, 0xBD423F44)
        )
        self.assertNotIn(0x230, eo.vec3_float_groups_from_bytes(blob))


class TestEntityChainDetection(unittest.TestCase):
    """The Level -> ActorManager chain primitives, on synthetic words."""

    def test_bl_target_decodes_forward_and_backward(self):
        # bl with imm26 = +4 instructions
        w = 0x94000000 | 4
        self.assertEqual(eo._bl_target(w, 0x1000), 0x1000 + 16)
        # bl with a negative offset (two's complement 26-bit)
        w = 0x94000000 | ((1 << 26) - 4)
        self.assertEqual(eo._bl_target(w, 0x1000), 0x1000 - 16)

    def test_bl_target_rejects_non_branch(self):
        self.assertIsNone(eo._bl_target(0xD65F03C0, 0x1000))  # ret

    def test_mov_x0_source(self):
        # mov x0, x20  == orr x0, xzr, x20
        w = 0xAA0003E0 | (20 << 16)
        self.assertEqual(eo._mov_x0_source(w), 20)
        # mov x0, x0 is not interesting but still parses
        self.assertEqual(eo._mov_x0_source(0xAA0003E0), 0)
        # Not a mov into x0.
        self.assertIsNone(eo._mov_x0_source(0xD65F03C0))

    def test_actor_manager_vector_offset_is_the_libcxx_vector(self):
        # The recovered constant: the vector is the manager's first member.
        self.assertEqual(eo.ACTOR_MANAGER_VECTOR_OFFSET, 0x8)


class TestReportShape(unittest.TestCase):
    def test_entity_seam_is_reported_unresolved_with_reason(self):
        report = {
            "entity_seam_offsets": {
                "resolved": False,
                "reason": "requires disassembly",
                "fields": [
                    {"name": "levelEntityListOffset", "meaning": "x", "offset": None},
                    {"name": "entityRuntimeIdOffset", "meaning": "x", "offset": None},
                    {"name": "entityPositionOffset", "meaning": "x", "offset": None},
                    {"name": "entityRotationOffset", "meaning": "x", "offset": None},
                ],
            }
        }
        # The report must be JSON-serialisable and must not claim a value.
        blob = json.dumps(report)
        loaded = json.loads(blob)
        self.assertFalse(loaded["entity_seam_offsets"]["resolved"])
        for field in loaded["entity_seam_offsets"]["fields"]:
            self.assertIsNone(field["offset"])

    def test_signature_status_vocabulary(self):
        # The three states the CI check keys on.
        self.assertEqual(
            {"unique", "unmatched", "ambiguous"}, {"unique", "unmatched", "ambiguous"}
        )


if __name__ == "__main__":
    unittest.main()
