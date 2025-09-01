# Docker 
---
### What is the meaning of docker?
Docker is an open-source platform that enables developers to build, deploy, run, update and manage containers.

Containers are standardized, executable components that combine application source code with the operating system (OS) libraries and dependencies required to run that code in any environment.


### Why we need Docker ?
- Consistency across platforms :- 
Docker is needed because software doesn’t just consist of code — it also depends on the environment it runs in (OS, libraries, dependencies, configurations). Normally, "it works on my machine" becomes a huge problem when running the same code on different systems.

   - Without Docker: You may have Python 3.12, but your teammate has Python 3.8 → things break.

  - With Docker: Everyone runs the same container with the same dependencies → no conflicts.

- Isolation :- means that each container runs like it has its own environment, separate from others, even though they’re all on the same machine.    
   -  Each container has its own filesystem, dependencies, libraries, and network ports. 
   -  If one container crashes, it doesn’t bring down others.
   - If one app needs Python 2.7 and another needs Python 3.12, both can happily coexist without conflict.

- Scalability = the ability of an application to handle more load by adding more resources (usually more instances/containers).
   - If 10 users are using your app, 1 container may be enough.
   - If 10,000 users come, you can spin up 10 containers to share the load.
   - Docker makes this process fast and automated.
---                         
# Docker Engine  
Docker Engine is the core software that runs and manages containers on your machine.
Docker Engine has 3 main parts:
- Docker Daemon (dockerd)
   - It runs on host machine.
   - Background service that does all the heavy lifting.
   - Creates, runs, and manages containers.
   - Listens for API requests (from the CLI or REST API).
- Docker CLI (docker command)
  - What you interact with.
  - When you type commands like docker run, docker build, docker ps, the CLI talks to the daemon.
- REST API
  - The interface between CLI (or other tools) and the daemon.
  - You could even use the API directly instead of CLI.
---
# Docker Images
A Docker image is a read-only blueprint/template used to create containers. It contains everything your application needs to run:
- Application code
- Runtime (like Python, Node.js, Java, etc.)
- System libraries
- Configurations
 
 **Life Cycle Of Docker Images**     
 1. Create an image
 2. Store 
 3. Distribute
 4.Execute
---
 # Docker Containers
 A container is a running instance of a Docker image.
It has its own isolated environment, including:
- File system
- Libraries and dependencies
- Network ports
- Processes
- Containers are lightweight, portable, and temporary.
- You can start, stop, restart, and delete them without affecting other containers.
---
# Docker Registry
A Docker Registry is a storage and distribution system for Docker images.
> It’s like a library or warehouse where images are stored and shared.(like ***GitHub***)     

You can push your images to a registry and pull them from anywhere.
### Types of Registries
1. Docker Hub → Public registry.
2. Private Registries → Hosted on your own server or cloud.
### Key differences between GitHub and Docker Hub
- Multiple Repositories:
   - GitHub: Each repository is usually a different project.
   - Docker Hub: Multiple repositories can exist, but each repository is usually a single app, and inside it, multiple tags/versions exist.
- Versioning inside repository:
   - GitHub → branches & commits track code evolution.
   - Docker Hub → tags track different image versions for the same app. 

### Types of Docker Registries
1. Docker Hub
 - Public registry managed by Docker.
- Most commonly used for open-source images.
- Anyone can pull public images, but pushing requires an account.
2. Private Registry
- Hosted by your organization or yourself.
- Restricted access → only specific users can push/pull images.
- Use case: Storing internal company images, sensitive apps, or proprietary software.
3. Third-Party Registry
- Managed by external providers, often offering advanced features.
- Usually comes with extra security, CI/CD integration, analytics.
- Enterprise deployment, multi-cloud environments, advanced workflow needs.

### Benefits of Registry:- 
- Centralized Storage
  - Stores all your Docker images in one place.
   - Easy to manage and organize multiple images and versions.
- Version Control
  - Images can have tags for different versions (latest, v1.0, etc.).
  - Makes it easy to roll back to a previous version if needed.
- Easy Sharing and Collaboration
  - Teams can push and pull images from the registry.
  - Ensures everyone works with the same environment.
- Portability
  - Images from the registry can be run anywhere Docker is installed.
  - Makes deploying applications to different environments (dev, staging, production) seamless.
- Integration with CI/CD
  - Registries integrate with pipelines to automate build, test, and deployment.
  - Supports automated image updates.
- Security and Access Control
  - Private registries allow restricted access.
  - Control who can push or pull images, ensuring sensitive images stay secure.
- Efficiency
  - Layers in images are reused → saves storage space.
  - Pulling images only downloads updated layers, reducing bandwidth usage.

  ---
  # Docker File Template

\#1. Choose a base image (eg python)

\# 2. Set a working directory inside the container
> WORKDIR /app

\# 3. Copy dependency file first (for caching layers)
> COPY requirements.txt .

\# 4. Install dependencies
> RUN pip install --no-cache-dir -r requirements.txt

\# 5. Copy the rest of your application code
> COPY . .

\# 6. Expose port (if your app runs on a port, e.g., 8000 for FastAPI/Flask)
> EXPOSE 8000

\# 7. Run the application
> CMD ["python", "main.py"]
