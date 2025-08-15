pipeline{
    agent any 

    environment{
        VENV_DIR = 'venv'
        GCP_PROJECT = "elegant-expanse-468605-g9"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
    }

    stages{
        stage('Cloning github repo to Jenkins.'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins .....................'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/adityach007/Hotel-Reservation-Prediction.git']])
                }
            }
        }
    }
}