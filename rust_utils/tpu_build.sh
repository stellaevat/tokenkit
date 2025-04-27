# otherwise use 'maturin develop --release'
set -e

. ../tokenkit_env/bin/activate
python3 -m maturin build --release
mv $PWD/target/wheels/rust_utils-0.14.1.dev0-cp310-cp310-manylinux_2_34_x86_64.whl $PWD/target/wheels/rust_utils-0.14.1.dev0-cp310-none-any.whl
pip install --force-reinstall $PWD/target/wheels/rust_utils-0.14.1.dev0-cp310-none-any.whl
#pip install fsspec==2023.9.2
#pip install --upgrade huggingface_hub datasets
# TODO: do we need the above? if yes, pin versions / check why