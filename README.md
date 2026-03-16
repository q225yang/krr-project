# krr-project

## How to use Sol Cluster + VScode

- From local machine, ssh to Sol cluster in **Terminal**:

    ```
    ssh [asuid]@login.sol.rc.asu.edu
    ```

- Verify the Duo Authantication and prompt the password.

- After logging into the cluster, clone the repo (you might need to set up the ssh key for the first time, you can follow the github instructions) and setup the environment.

    ```
    git clone git@github.com:q225yang/krr-project.git
    chmod +x scripts/setup_env.sh
    scripts/setup_env.sh        # Also works for updating the environment
    ```
    The conda environment is called `krr`.

    If you need to update the environment, you can add the new packages to environment.yml and rerun the script to update the env.

- To start a VSCode server with GPU using vscode_starter.sh. (Details can be found at https://asurc.atlassian.net/wiki/spaces/RC/pages/1907818602/VSCode#Create-a-VSCode-Tunnel-Via-the-Command-line)

    ```
    chmod +x scripts/vscode_starter.sh
    scripts/vscode_starter.sh
    # scripts/vscode_starter.sh 0-12 # Allocate a 12-hour vscode session
    ```
    Then, you can follow the prompt to set up the tunnel and connect to the tunnel in your local VSCode.

    **Don't simply use ssh command to connect to the cluster in VSCode, or you will receive usage warnings.**

## Download the Datasets

Details: https://huggingface.co/datasets/USC-GVL/PhysBench

```
mkdir /scratch/[asuid]/krr-data
cd /scratch/[asuid]/krr-data
huggingface-cli download USC-GVL/PhysBench --local-dir . --local-dir-use-symlinks False --repo-type dataset
yes | unzip image.zip -d image
# yes | unzip video.zip -d video
```

The answer file is at: https://github.com/physical-superintelligence-lab/PhysBench/blob/main/eval/physbench/test_answer.json

##
