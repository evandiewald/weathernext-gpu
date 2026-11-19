"""CPU check that chunked_graph.py computes the same forecast as the port's original code.

Small random model (mesh_splits=2, 684 grid points), random receiver-sorted geometry with
uneven fan-in (some mesh nodes receive no edges), several chunk sizes. Runs in seconds on a
laptop, no GPU or downloads:
    python scripts/test_chunked_graph.py
Expect: float64 max |diff| 0 (or ~1e-15) at every chunk size; fp32 ~1e-7 (rounding only).
"""
import copy
import pathlib
import sys

import torch
from transformers import WeatherNext2Config, WeatherNext2ForWeatherForecasting

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from chunked_graph import install_chunked  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg = WeatherNext2Config(mesh_splits=2, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, grid_latitudes=19, grid_longitudes=36,
                             num_grid_to_mesh_edges=2048, attention_bandwidth=64,
                             graph_config={"hidden_size": 32})
    ref = WeatherNext2ForWeatherForecasting(cfg).double().eval()
    geo = ref.model.geometry
    g, m, e = cfg.num_grid_points, cfg.num_mesh_nodes, cfg.num_grid_to_mesh_edges
    print(f"grid {g}, mesh {m}, grid->mesh edges {e}, mesh->grid edges {3 * g}")
    geo.grid_to_mesh_receivers.copy_(torch.sort(torch.randint(0, m - 5, (e,))).values)
    geo.grid_to_mesh_senders.copy_(torch.randint(0, g, (e,)))
    geo.mesh_to_grid_senders.copy_(torch.randint(0, m, (3 * g,)))
    for name in ("grid_to_mesh_edge_features", "mesh_to_grid_edge_features",
                 "grid_spatial_features", "mesh_spatial_features"):
        getattr(geo, name).normal_()
    for p in ref.parameters():          # default init leaves some layers at zero; make every path matter
        p.data.normal_(0, 0.3)

    b = 2
    blocks = -(-m // cfg.attention_bandwidth)
    inputs = dict(
        grid_features=torch.randn(b, cfg.num_grid_input_channels - 3, 19, 36, dtype=torch.float64),
        global_features=torch.randn(b, cfg.num_mesh_input_channels - 3, dtype=torch.float64),
        noise=torch.randn(b, cfg.noise_channels, dtype=torch.float64),
        attention_mask=torch.ones(blocks, 1, cfg.attention_bandwidth, 3 * cfg.attention_bandwidth,
                                  dtype=torch.bool))
    with torch.inference_mode():
        want = ref(**inputs).prediction
        for chunk in (7, 50, 333, 10**9):
            model = copy.deepcopy(ref)
            install_chunked(model, chunk=chunk)
            got = model(**inputs).prediction
            rel = ((got - want).norm() / want.norm()).item()
            print(f"float64 chunk {chunk:>10}: max |diff| {(got - want).abs().max().item():.2e}  rel {rel:.2e}")
            assert rel < 1e-12, rel
        f32 = {k: (v.float() if v.is_floating_point() else v) for k, v in inputs.items()}
        ref32 = copy.deepcopy(ref).float()
        want32 = ref32(**f32).prediction
        m32 = copy.deepcopy(ref32)
        install_chunked(m32, chunk=50)
        rel = ((m32(**f32).prediction - want32).norm() / want32.norm()).item()
        print(f"fp32    chunk         50: rel {rel:.2e}")
        assert rel < 1e-5, rel
    print("OK")


if __name__ == "__main__":
    main()
