from enum import Enum

# list of initial noises randomly sampled from https://gitlab-master.nvidia.com/earakelyan/adversarial_reasoning

class NoiseType(Enum):
    # MATH_FACT = "For odd x, x^2 mod 8=1."  # 13 tokens
    # CODE_SNIPPET = "Tensor_3D = torch.diag_embed(Tensor_2D)"  # 14 tokens
    # RANDOM_FACT = "The song remained a karaoke staple in Korea for many years."  # 13 tokens
    # MARKUP_NOISE = "<r><a>zo</a><b x=\"ab\"></b></r>"  # 17 tokens

    # MATH_FACT = "Lemma_3 ≈ diag⊗(Σ_2D) ⇒ ∀x∈∅"
    # CODE_SNIPPET = "fn song_remained := (kara0ke_staple[K0rea] ++ yrs_many); // for(;;) remain??"
    # RANDOM_FACT = "Octopus grow wings each winter."
    # MARKUP_NOISE = "##<odd_x> *x^2* ~~mod~~ 8==[1] </odd_x>"

    LONG_MATH_FACT = "(a+b) mod m=((a mod m)+(b mod m) mod m). (a-b) mod m=((a mod m)-(b mod m) mod m). ab mod m=((a mod m)*(b mod m) mod m). a^k mod m=((a mod m)^k mod m)." # 64
    LONG_CODE_SNIPPET = "def chess_knight(cell):\n    col, row = cell\n    col, row = ord(col) - 97, int(row) - 1\n    return sum(abs(y-row)**2 + abs(x-col)**2 == 5 for y in range(8) for x in range(8))" # 64
    LONG_RANDOM_FACT = "Walt Disney Pictures released it in the United States and Canada on June 27, 2008, grossing $23.1\u00a0million on its opening day, and $63\u00a0million during its opening weekend in 3,992 theaters, ranking number 1 at the box office." # 65
    LONG_MARKUP_NOISE = "```xml\n<root>\n  <alpha id=\"xy\">foo</alpha>\n  <beta>\n    <gamma>bar</gamma>\n    <delta code=\"abc\"/>\n  </beta>\n  <epsilon/>\n  <zeta>baz</zeta>\n  <theta tag=\"mn\"/>\n</root>\n```" # 65




