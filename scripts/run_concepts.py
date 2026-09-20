from kube_jobs import storage, submit_job


submit_job(
    job_name="xai-framework-concept-bounds",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="16Gi",
    gpu=1,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/project_name workdir",
        "cd workdir",
        "uv sync",
        "uv run -m concepts.bounds model=resnet50 data=imagenette method=pca",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
