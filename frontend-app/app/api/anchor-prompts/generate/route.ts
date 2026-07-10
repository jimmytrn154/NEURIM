import { NextResponse } from "next/server";
import OpenAI from "openai";

type GenerateRequest = {
  desired_prompt?: string;
};

function normalizePrompts(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean).slice(0, 10);
}

function normalizeAxes(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean).slice(0, 5);
}

export async function POST(request: Request) {
  let body: GenerateRequest;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON request body" }, { status: 400 });
  }

  const desired = body.desired_prompt?.trim();
  if (!desired) {
    return NextResponse.json({ error: "desired_prompt is required" }, { status: 400 });
  }

  if (!process.env.OPENAI_API_KEY) {
    return NextResponse.json({ error: "OPENAI_API_KEY is not configured for frontend-app" }, { status: 500 });
  }

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const model = process.env.OPENAI_PROMPT_MODEL ?? "gpt-5.5";

  try {
    const metaPrompt = `You are generating anchor prompts for a real-time latent-space image morphing system.

User prompt:
"${desired}"

Task:
Generate a bank of exactly 10 anchor prompts derived from the user prompt. The prompts will be converted into embeddings, reduced using PCA, and used as anchor points for smooth latent-space interpolation.

The anchor bank must preserve the central semantic identity and visual intent of the user prompt while introducing controlled, coherent visual variation. The prompts should create useful latent directions without causing abrupt changes in subject, scene, composition, or artistic identity.

Instructions:

1. Establish a stable visual scaffold from the user prompt. The scaffold may include:

   * Main subject or subjects
   * Environment or background
   * Camera distance and viewing angle
   * Composition and spatial arrangement
   * Lighting setup
   * Visual medium or artistic style
   * Overall mood or atmosphere


3. Identify 2-4 controlled variation axes that are:

   * Visually meaningful
   * Compatible with the original prompt
   * Likely to produce smooth interpolation
   * Independent enough to create useful latent directions

4. Suitable variation axes may include:

   * Color or hue
   * Material
   * Surface texture
   * Shape or proportion
   * Pose or orientation
   * Facial expression
   * Age or degree of transformation
   * Object density
   * Pattern complexity
   * Motion intensity
   * Weather intensity
   * Lighting intensity or temperature
   * Mood intensity
   * Degree of realism
   * Environmental state
   * Abstract visual properties explicitly implied by the prompt

6. Keep all non-varied attributes as consistent as possible.

7. Do not introduce:

   * New subjects
   * New characters
   * New objects
   * New locations
   * Unrelated actions
   * Different camera compositions
   * Different visual styles
   * Major semantic changes

   An exception is allowed only when the original user prompt explicitly requests variation in those elements.


9. Write each anchor prompt as a complete, self-contained image-generation prompt. Do not use shorthand such as "same image but blue" or refer to another anchor prompt.


11. Avoid extreme endpoints that could break subject identity, composition, or visual continuity.

12. When the user prompt is vague:

* Infer the simplest stable visual scaffold.
* Do not add elaborate narrative details.
* Prefer safe variations in color, material, texture, shape, pose, expression, density, atmosphere, or lighting intensity.
* Keep inferred details consistent across every anchor.

13. When the user prompt already contains several changing elements, select only the most visually useful and interpolation-friendly axes. Treat the remaining elements as fixed.

14. Do not include explanations, commentary, numbering, interpolation values, or parameter syntax inside the anchor prompts.

Output requirements:

* Return exactly 10 anchor prompts.

* Each anchor prompt must contain exactly one sentence.

* Place every anchor prompt inside quotation marks.

* Format every anchor prompt using exactly this prefix:

* * "prompt example"

Output format:


Anchor prompts:

* * "[Anchor prompt 1]"
* * "[Anchor prompt 2]"
* * "[Anchor prompt 3]"
* * "[Anchor prompt 4]"
* * "[Anchor prompt 5]"
* * "[Anchor prompt 6]"
* * "[Anchor prompt 7]"
* * "[Anchor prompt 8]"
* * "[Anchor prompt 9]"
* * "[Anchor prompt 10]"

Before producing the output, silently verify that:

* There are exactly 10 anchor prompts.
* Every prompt preserves the same core subject and intent.
* No unrequested objects, characters, scenes, or styles have been introduced.
* The anchor bank supports gradual and visually coherent interpolation.
`;

    const response = await client.responses.create({
      model,
      input: [
        {
          role: "system",
          content:
            "You are the NEURIM anchor-prompt generation agent. Follow the user-provided meta-prompt exactly. Return machine-readable JSON matching the supplied schema while preserving the requested controlled-axes and anchor-prompts content.",
        },
        { role: "user", content: metaPrompt },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "anchor_prompt_set",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["controlled_axes", "anchor_prompts"],
            properties: {
              controlled_axes: {
                type: "array",
                minItems: 1,
                maxItems: 5,
                items: {
                  type: "string",
                  minLength: 3,
                },
              },
              anchor_prompts: {
                type: "array",
                minItems: 10,
                maxItems: 10,
                items: {
                  type: "string",
                  minLength: 12,
                },
              },
            },
          },
        },
      },
    });

    const parsed = JSON.parse(response.output_text || "{}") as { controlled_axes?: unknown; anchor_prompts?: unknown };
    const controlledAxes = normalizeAxes(parsed.controlled_axes);
    const anchorPrompts = normalizePrompts(parsed.anchor_prompts);
    if (anchorPrompts.length !== 10) {
      return NextResponse.json({ error: "OpenAI did not return exactly 10 anchor prompts" }, { status: 502 });
    }
    return NextResponse.json({ controlled_axes: controlledAxes, anchor_prompts: anchorPrompts, model });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
