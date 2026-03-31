DISTRACTOR_TYPES = [
    "RANDOM_FACT",
    "CODE_SNIPPET",
    "ENCRYPTED_TEXT",
    "MARKUP_NOISE",
    "MATH_FACT",
]

# Positions must match the numeric suffixes of prompts/eval-prompt-{N}.yaml.
# Position 0: noise BEFORE question  ({noise} {question_no_options} {options})
# Position 1: noise AFTER question   ({question_no_options} {noise} {options})
# Position 2: noise AFTER options    ({question_no_options} {options} {noise})
#
# Distractor texts are sampled from real adversarial_reasoning datasets,
# targeting 12-17 Qwen tokens (text_length key). MARKUP_NOISE has no short
# examples; shortest available were used instead.
INITIAL_DISTRACTORS = [
    # ── RANDOM_FACT ─────────────────────────────────────────────────────────
    {
        "id": "0_RANDOM_FACT_0",
        "distractor": "The Reign originally played in the Women's Professional Football League.",
        "type": "RANDOM_FACT",
        "position": 0,
    },
    {
        "id": "1_RANDOM_FACT_1",
        "distractor": "The screenplay was written by Barry Wernick and James R. Hallam.",
        "type": "RANDOM_FACT",
        "position": 1,
    },
    {
        "id": "2_RANDOM_FACT_2",
        "distractor": "She was occasionally credited under her full name, Mary Anita Loos.",
        "type": "RANDOM_FACT",
        "position": 2,
    },
    # ── CODE_SNIPPET ─────────────────────────────────────────────────────────
    {
        "id": "3_CODE_SNIPPET_0",
        "distractor": "def summy(s):\n    return sum(map(int, s.split(' ')))",
        "type": "CODE_SNIPPET",
        "position": 0,
    },
    {
        "id": "4_CODE_SNIPPET_1",
        "distractor": "Feature = sparse.hstack((c1, c2)).tocsr()",
        "type": "CODE_SNIPPET",
        "position": 1,
    },
    {
        "id": "5_CODE_SNIPPET_2",
        "distractor": "def isDigit(string):\n    return string.replace('.', '').strip('-').isdigit()",
        "type": "CODE_SNIPPET",
        "position": 2,
    },
    # ── ENCRYPTED_TEXT ───────────────────────────────────────────────────────
    {
        "id": "6_ENCRYPTED_TEXT_0",
        "distractor": "Jwh qdn itqupzu Blmv Brgghhm zc wbh hlhlwdg rcg jrmkgdshy sn Guugvch Frqv.",
        "type": "ENCRYPTED_TEXT",
        "position": 0,
    },
    {
        "id": "7_ENCRYPTED_TEXT_1",
        "distractor": "Bktb temkktjutm gpy jdljkg emfqia uw svbbug pim dnzuddnniz bz wecoqqfnks jwapzwra.",
        "type": "ENCRYPTED_TEXT",
        "position": 1,
    },
    {
        "id": "8_ENCRYPTED_TEXT_2",
        "distractor": "Kht O. Looovw Iacrksvk Phdpjg Dpmk Fjyodz ez g ibbi-yton waeypt hxud zikbvc ic Bad Erer Tiim.",
        "type": "ENCRYPTED_TEXT",
        "position": 2,
    },
    # ── MARKUP_NOISE ─────────────────────────────────────────────────────────
    {
        "id": "9_MARKUP_NOISE_0",
        "distractor": '<div class="abx">foo</div>\n<h3 id="pqr">bar</h3>\n<ul>\n  <li>baz</li>\n  <li></li>\n</ul>\n<p>qux</p>\n<span style="rgb">quz</span>',
        "type": "MARKUP_NOISE",
        "position": 0,
    },
    {
        "id": "10_MARKUP_NOISE_1",
        "distractor": '<h3 id="abc">zen fox</h3>\n<div class="xy">lorem</div>\n<p>oak</p>\n<ul>\n  <li>red</li>\n  <li id="qrs"></li>\n</ul>\n<span style="bg">sun</span>',
        "type": "MARKUP_NOISE",
        "position": 1,
    },
    {
        "id": "11_MARKUP_NOISE_2",
        "distractor": '<root>\n  <alpha id="xyz">foo</alpha>\n  <beta>\n    <gamma>bar</gamma>\n    <delta code="abc"/>\n  </beta>\n  <epsilon/>\n  <zeta>baz</zeta>\n  <theta tg="def"/>\n</root>',
        "type": "MARKUP_NOISE",
        "position": 2,
    },
    # ── MATH_FACT ────────────────────────────────────────────────────────────
    {
        "id": "12_MATH_FACT_0",
        "distractor": "sum_(k=1)^n C(n, k)=2^n.",
        "type": "MATH_FACT",
        "position": 0,
    },
    {
        "id": "13_MATH_FACT_1",
        "distractor": "For real numbers a, b, (a+b)^2\u22654ab.",
        "type": "MATH_FACT",
        "position": 1,
    },
    {
        "id": "14_MATH_FACT_2",
        "distractor": "tan(2a)=2tan(a)/(1-tan(a)^2).",
        "type": "MATH_FACT",
        "position": 2,
    },
]
