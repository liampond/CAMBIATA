TASK: Compose a counterpoint voice ABOVE the given cantus firmus in fifth species counterpoint (florid counterpoint — a free combination of whole notes, half notes, quarter notes, and syncopations). The counterpoint must be on STAFF 1 (top staff) and the cantus firmus on STAFF 2 (bottom staff).

VOICE PLACEMENT:
- The cantus firmus is the LOWER voice. You MUST compose the counterpoint as the UPPER voice.
- Assign treble clef to the counterpoint voice.
- Do NOT rewrite, alter, transpose, or re-voice the cantus firmus. It must appear in the output exactly as given.

STRUCTURE:
- Each measure may freely combine rhythmic values from all previous species: whole notes (1st), half notes (2nd), quarter notes (3rd), and syncopated/tied half notes (4th)
- Measure N (FINAL): exactly ONE whole note for the cadence
- Rhythmic variety is essential — avoid writing consecutive measures in the same species
- Suspensions (4th species) should be used where possible, especially approaching cadences

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
