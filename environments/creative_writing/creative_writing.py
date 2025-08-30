import verifiers as vf
import os
import openai
import json
import datasets

JUDGE_MODEL = "deepseek/deepseek-chat-v3.1"
PARSE_MODEL = "google/gemini-2.5-flash-lite"
client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

def judge_response(prompt, completion, answer, state):
    judge_prompt = f"""You will be serving as a judge for a very tough creative writing competition. You will be using a rubric to grade a story based on a prompt. Be mean, and do not give benefit of the doubt unless it is truly fantastic.

The rubric:

| Skill | 1 | 2 | 3 | 4 |
|-|-|-|-|-|
| **Instruction Following** | The story displays little, if any, relevance to what was asked for in the prompt. | The story follows some parts of the prompt, but ignores others | The story attempts to follow all parts of the prompt, but some parts are mostly ignored beyond a passing mention | The story follows all parts of the prompt as described. |
| **Subtlety** | Elements are clumsily forced in through direct copying or obvious paraphrasing from the prompt. Uses blatant "its not X; its Y" constructions or list-like exposition dumps. | Incorporates elements through close paraphrasing or obvious symbolic stand-ins. May use heavy-handed metaphorical statements that feel artificially inserted. | Mostly natural integration with occasional forced connections or slightly obvious symbolic elements. | All prompt elements flow naturally within the story's logic without obvious insertion or paraphrasing. |
| **Sentence Variety** | Extremely repetitive sentence patterns (e.g., all short fragments or all flowing complex sentences). Heavy reuse of identical phrase structures and openings. | Limited sentence variety with noticeable patterns. Some phrase repetition but attempts at variation. | Good sentence variety with occasional repetitive patterns. Mostly avoids repeated phrase structures. | Excellent variety in sentence length, structure, and rhythm. No noticeable repetitive patterns. |
| **Balance** | Over-reliance on single devices (excessive metaphor, constant anthropomorphization, repetitive imagery patterns). Creates exhaustion through monotonous technique use. | Limited range of literary devices with some over-use of preferred techniques. Occasional effective moments. | Good variety of literary devices with mostly effective deployment. Minor over-reliance on certain techniques. | Sophisticated, varied use of literary devices. Each technique serves a clear purpose and enhances the narrative. |
| **Authenticity** | Artificial, performative language, forced relatability, or robotic "follow orders" phrasing. | Mostly artificial voice with some genuine moments. May slip into overly enthusiastic or explanatory modes. | Generally authentic voice with occasional slips into artificial enthusiasm or over-explanation. | Consistent, believable narrative voice throughout. Feels genuinely human without performance or artificiality. |
| **Specificity** | Vague, abstract descriptions or inappropriate hyper-specificity (such as weirdly precise numbers). Heavy use of still life that adds no narrative value. | Mix of vague abstractions and occasionally meaningful details. Some irrelevant specificity. | Mostly meaningful, well-chosen details with occasional vague or overly precise moments. | All details serve narrative purpose. Specificity feels natural and enhances understanding without being arbitrary. |
| **Coherence** | Relies on "lolrandom" juxtapositions or glossolalia that prioritizes aesthetics over meaning. Incoherent or contradictory elements. | Some logical inconsistencies or random elements that don't serve the story. Occasional meaningful connections. | Generally coherent with minor logical gaps or slightly random elements that don't significantly detract. | Fully coherent internal logic. All elements connect meaningfully to create a unified whole. |
| **Originality & Freshness** | Heavy reliance on common writing patterns: nostalgia, dilapidation imagery, SCP-style concepts, or overused names (Elara, Nova, Chen). | Some original elements mixed with recognizable clichés and common tropes. | Mostly original with occasional familiar elements that don't dominate the narrative. | Genuinely fresh perspective and approach. Avoids common tropes and predictable patterns entirely. |
| **Emotional Resonance** | Emotions feel manufactured or performed rather than earned. May use cheap "#relatable" moments or forced universal experiences. | Some genuine emotional moments undermined by artificial or manipulative elements. | Generally authentic emotional content with minor forced or unearned moments. | Emotions arise naturally from character and situation. Resonates authentically without manipulation. |

Prompt:
```
{prompt}
```

Story:
```
{completion}
```"""
    res = client.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": judge_prompt}])
    parse_prompt = f"""Extract the scores from the following response:
```
{res.choices[0].message.content}
```

Put them into a json that looks like:
```
{{
    "instruction_following": int,
    "subtlety": int,
    "sentence_variety": int,
    "balance": int,
    "authenticity": int,
    "specificity": int,
    "coherence": int,
    "originality_freshness": int,
    "emotional_resonance": int
}}
```
Do not output a code block, it will be directly fed to a JSON parser"""
    res = client.chat.completions.create(model=PARSE_MODEL, messages=[{"role": "user", "content": parse_prompt}])
    scores = json.loads(res.choices[0].message.content)
    average = sum(scores.values()) / len(scores)
    return average


def load_environment(**kwargs) -> vf.Environment:
    '''
    Loads a custom environment.
    '''

    dataset = datasets.load_dataset("allura-org/fujin-instruct-v2", split="train")
    dataset = dataset.map(lambda x: {"prompt": [{"role": "user", "content": x["conversations"][0]["value"].strip('"')}], "info": {"answer": ""}}).remove_columns(["conversations"])

    rubric = vf.Rubric(funcs=[judge_response], weights=[1.0])

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt="You are a creative writer participating in a creative writing contest.",
        rubric=rubric,
        **kwargs
    )
