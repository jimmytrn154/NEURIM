import numpy as np
from PIL import Image

from src.generator.openai_image import OpenAIImageGenerator
from src.generator.procedural import ProceduralRenderer
from src.generator.to_3d import ProceduralPseudo3D, mirrored_quadrants
from src.optimizer.projection import AnchorInterpolationProjector, PCAProjector


def test_procedural_renderer_produces_image_of_requested_size():
    renderer = ProceduralRenderer()
    img = renderer.render(np.array([0.1, -0.2, 0.5, 0.0, 0.3]), size=128)
    assert isinstance(img, Image.Image)
    assert img.size == (128, 128)


def test_procedural_renderer_changes_with_z():
    renderer = ProceduralRenderer()
    img_a = renderer.render(np.array([-1.0, -1.0, -1.0, 0.0, 0.0]), size=64)
    img_b = renderer.render(np.array([1.0, 1.0, 1.0, 0.0, 0.0]), size=64)
    assert list(img_a.getdata()) != list(img_b.getdata())


def test_mirrored_quadrants_composes_full_canvas():
    renderer = ProceduralRenderer()
    img = renderer.render(np.array([0.2, 0.4, 0.1]), size=64)
    canvas = mirrored_quadrants(img, canvas_size=200)
    assert canvas.size == (200, 200)


def test_pseudo_3d_squashes_at_90_degrees():
    renderer = ProceduralRenderer()
    img = renderer.render(np.array([0.2, 0.4, 0.1]), size=64)
    pseudo3d = ProceduralPseudo3D()
    import math

    front = pseudo3d.apply_angle(img, 0.0)
    side = pseudo3d.apply_angle(img, math.pi / 2)
    assert front.size == side.size == img.size


def test_pca_projector_roundtrip():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(6, 40))
    projector = PCAProjector(dims=4).fit(embeddings)
    z = np.array([0.1, -0.2, 0.05, 0.0])
    embedding = projector.to_embedding(z)
    assert embedding.shape == (40,)


def test_anchor_projector_stays_in_convex_hull():
    anchors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    projector = AnchorInterpolationProjector(anchors)
    embedding = projector.to_embedding(np.array([0.0, 0.0, 0.0]))
    # Uniform softmax weights -> centroid of the anchors.
    assert np.allclose(embedding, anchors.mean(axis=0), atol=1e-6)


class _FakeImages:
    def __init__(self, b64_json: str):
        self.b64_json = b64_json
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        image = type("ImageResult", (), {"b64_json": self.b64_json})()
        return type("ImageResponse", (), {"data": [image]})()


class _FakeOpenAIClient:
    def __init__(self, b64_json: str):
        self.images = _FakeImages(b64_json)


def test_openai_image_generator_decodes_and_caches_prompt():
    import base64
    import io

    src = Image.new("RGB", (16, 16), (255, 0, 0))
    buf = io.BytesIO()
    src.save(buf, format="PNG")
    client = _FakeOpenAIClient(base64.b64encode(buf.getvalue()).decode("ascii"))
    generator = OpenAIImageGenerator(frame_size=8, client=client)

    img_a = generator.render_prompt("a red square")
    img_b = generator.render_prompt("a red square")

    assert img_a.size == (8, 8)
    assert img_b.size == (8, 8)
    assert len(client.images.calls) == 1
    assert client.images.calls[0]["model"] == "gpt-image-2"
