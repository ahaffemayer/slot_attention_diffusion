# Conditional Diffusion Motion

This repository contains the code and resources for the paper submission.

## Dataset and Resources
The full dataset and additional resources are available via OSF at the following **anonymous link**:

🔗 [Download resources (OSF)](https://osf.io/jsr7u/?view_only=92190681e4134f6798f1105ff63f7b56)

For direct command-line access:

```bash
 curl -L "https://osf.io/download/jyvfa/?view_only=92190681e4134f6798f1105ff63f7b56" -o ressources.zip 

```

Once downloaded, extract the contents with:

```bash
unzip ressources.zip
```

## Code
The core implementation of Conditional Diffusion Motion is included in this repository.

## Example
To run the example with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv run examples/main_shelf_slot_attention.py
```
and
```bash
uv run meshcat-server
```

And open the meschat link on your webbrowser.

## Docker

A Dockerfile is also provided. To use it:

```bash
docker build -t conditional_diffusion_motion .
docker run -p 7000:7000 --rm -it conditional_diffusion_motion
```
