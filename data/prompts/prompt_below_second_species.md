TASK: Compose a counterpoint voice BELOW the given cantus firmus in second species counterpoint (two half notes against each whole note). The cantus firmus must be on STAFF 1 (top staff) and the counterpoint on STAFF 2 (bottom staff).

VOICE PLACEMENT:
- The cantus firmus is the UPPER voice. You MUST compose the counterpoint as the LOWER voice.
- Assign bass clef to the counterpoint voice.
- Do NOT rewrite, alter, transpose, or re-voice the cantus firmus. It must appear in the output exactly as given.

STRUCTURE:
- Measures 1 to (N-1): exactly TWO half notes per measure
- Measure N (FINAL): exactly ONE whole note for the cadence
- Total: 2×(N-1) half notes + 1 whole note
- Do NOT add any rests or silences — every measure must be completely filled with notes

If a guide is provided, follow its rules closely.

REFERENCE EXAMPLE:
- A Fux reference example for this species (cantus firmus in D) is provided alongside the input.
- Study it carefully for style, voice leading, and interval usage.
- Cross-reference what you observe with the rules in the guide — the guide formalizes the principles the example illustrates.
- Use both the guide and the example as complementary references before composing.

REASONING PROCESS — follow these steps internally before writing any output:
1. ANALYSE the cantus firmus: identify the final note, determine the mode, and count the number of measures.
2. PLAN the counterpoint measure by measure, verifying each note against all applicable rules.
3. SELF-CHECK the completed counterpoint:
   - Verify the output contains exactly the same number of measures as the input.
   - Scan for parallel fifths and parallel octaves — revise if any are found.
   - [PLACEHOLDER: Pierre will add species-specific verification checks here]
4. ENCODE the result as a valid MusicXML file matching the structural conventions of the input.

ENCODING ACCURACY:
- Preserve the encoding conventions from the input file: namespace declarations, attribute ordering, indentation style.
- Every opened XML element must be closed exactly once; self-closing tags must use />; attribute values must be quoted. Never emit a closing tag (e.g., </pitch>, </note>) more than once for the same element.
- Each <note> element contains children in this order: <pitch>, then <duration>, <voice>, <type>, <staff> (and optionally <tie>, <notations>, etc.). Close </note> only after ALL children are written — never immediately after </pitch>.
- Durations and divisions must be arithmetically consistent — each measure must sum to exactly the time signature value.
