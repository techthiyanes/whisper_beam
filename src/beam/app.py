import beam

app = beam.App(
    name="hello-world",
    cpu=8,
    memory="32Gi",
    gpu="T4",
    python_version="python3.8",
    python_packages=[
        "numpy", "openai-whisper", "scipy", "ftfy", "torch"
    ],
    commands=["apt-get update && apt-get install -y ffmpeg"]
    # commands=[
    #     "apt update && apt upgrade -y && apt install g++-8 gcc-8 -y",
    #     "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 700 --slave /usr/bin/g++ g++ /usr/bin/g++-7",
    #     "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8",
    #     "git clone https://github.com/ggerganov/whisper.cpp.git && cd whisper.cpp && bash ./models/download-ggml-model.sh tiny && make"
    # ]
)

app.Trigger.RestAPI(
    inputs={"audio": beam.Types.Json()},
    outputs={
        "response": beam.Types.String(),
    },
    handler="run.py:process_audio",
    loader="run.py:load_processor_model"
)

app.Mount.PersistentVolume(path="./cache_data", name="cache_data")