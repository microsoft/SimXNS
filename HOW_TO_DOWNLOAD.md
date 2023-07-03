# How to Download Desired Resources

Since most of the resources are large, we release the preprocessed data and trained ckpts in an Azure Blob container with public access for the most of our open-source projects.
We here provide two methods to download the resources.


## You Must Know

In almost each projects, we will provide the **basic URL** of the project in the `README.md` file.
For example, for [SimANS](https://github.com/microsoft/SimXNS/tree/main/SimANS), the basic URL is `https://msranlciropen.blob.core.windows.net/simxns/SimANS/`.

If you want to view the content of the blob, you can use [Microsoft's AzCopy CLI tool](https://learn.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy):
```bash
azcopy list https://msranlciropen.blob.core.windows.net/simxns/SimANS/
```
We also provide the list in the `README.md` file of projects.


## Method 1: Directly Download using the URLs

You may choose to download only a part of the resources by appending the relative path to the blob url and directly downloading it.


## Method 2: Faster Transmission using AzCopy

You can also use [the copy function of Microsoft's AzCopy CLI tool](https://learn.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-copy).
Here we provide the command of copying the entire folder like this:
```bash
azcopy copy --recursive "https://msranlciropen.blob.core.windows.net/simxns/SimANS/" .
```
You may also use the tool to copy only a single file as you like:
```bash
azcopy copy "https://msranlciropen.blob.core.windows.net/simxns/SimANS/best_simans_ckpt/TQ/checkpoint-10000" .
```
