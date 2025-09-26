# Conditional Diffusion Motion

This repository contains the code and resources for the paper submission.

## Dataset and Resources
The full dataset and additional resources are hosted on OSF and accessible through an anonymous review link:

🔗 [Download resources (OSF)](https://osf.io/jsr7u/?view_only=92190681e4134f6798f1105ff63f7b56)

For direct command-line download:

```bash
curl -L "https://osf.io/download/jyvfa/?view_only=92190681e4134f6798f1105ff63f7b56" -o resources.zip
unzip resources.zip
```

## Code
The core implementation of Conditional Diffusion Motion is included in this repository.

## Example
To run the example with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv run examples/main_shelf_slot_attention.py
```
In a separate terminal, start the meshcat server:
```bash
uv run meshcat-server
```
Then open the link printed by `meshcat-server` (e.g. `http://127.0.0.1:7000`) in your browser to visualize the results.

And open the meschat link on your web browser.

If you've never used uv:
```bash
pip install uv
```

## Docker

A Dockerfile is also provided. To use it:

```bash
docker build -t conditional_diffusion_motion .
docker run -p 7000:7000 --rm -it conditional_diffusion_motion
```
