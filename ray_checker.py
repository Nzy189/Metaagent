import ray
print("cluster:", ray.cluster_resources())
print("available:", ray.available_resources())
