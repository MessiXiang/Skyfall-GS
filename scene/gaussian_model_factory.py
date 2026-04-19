from plyfile import PlyData

from scene.gaussian_model import GaussianModel as GaussianModel3D
from scene.gaussian_model_2d import GaussianModel2D


GAUSSIAN_BACKENDS = {
    "3dgs": GaussianModel3D,
    "2dgs": GaussianModel2D,
}


def normalize_gaussian_backend(backend_name):
    backend = (backend_name or "3dgs").strip().lower()
    aliases = {
        "3d": "3dgs",
        "3dgs": "3dgs",
        "mip-splatting": "3dgs",
        "2d": "2dgs",
        "2dgs": "2dgs",
        "surfel": "2dgs",
        "surfel-gs": "2dgs",
    }
    if backend not in aliases:
        raise ValueError(f"Unsupported gaussian backend: {backend_name}")
    return aliases[backend]


def get_gaussian_model_class(backend_name):
    return GAUSSIAN_BACKENDS[normalize_gaussian_backend(backend_name)]


def create_gaussian_model(backend_name, sh_degree, appearance_enabled, appearance_n_fourier_freqs, appearance_embedding_dim):
    model_class = get_gaussian_model_class(backend_name)
    return model_class(sh_degree, appearance_enabled, appearance_n_fourier_freqs, appearance_embedding_dim)


def create_gaussian_model_from_dataset(dataset):
    return create_gaussian_model(
        getattr(dataset, "gs_backend", "3dgs"),
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )


def detect_gaussian_backend_from_ply(ply_path):
    plydata = PlyData.read(ply_path)
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    if len(scale_names) == 2:
        return "2dgs"
    return "3dgs"