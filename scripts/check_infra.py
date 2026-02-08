import mlflow
import os
import time
import random
from dotenv import load_dotenv 
from mlflow.tracking import MlflowClient

load_dotenv(override=True) 

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

def run_check():
    print("[INFO] Starting connection check...")

    # Step 1: Create Experiment (Test Database Postgres)
    experiment_name = "System_Health_Check"
    client = MlflowClient()
    existing_exp = client.get_experiment_by_name(experiment_name)
    
    if existing_exp:
        if existing_exp.lifecycle_stage == 'deleted':
            client.restore_experiment(existing_exp.experiment_id)
    
    mlflow.set_experiment(experiment_name)

    # Step 2: Start Run & Log Data
    with mlflow.start_run(run_name=f"Test_Run_{int(time.time())}") as run:
        print(f"[INFO] Run ID: {run.info.run_id}")

        # Test Log Params/Metrics
        mlflow.log_param("check_type", "smoke_test")
        mlflow.log_metric("fake_accuracy", random.random())
        print("[OK] [Database] Logged params and metrics successfully.")

        # Step 3: Test Upload File (Test MinIO S3)
        filename = "test_artifact.txt"
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write("This is a test file for MinIO connection.\n")
            f.write(f"Timestamp: {time.time()}")
        
        try:
            mlflow.log_artifact(filename)
            print("[OK] [Storage] Uploaded artifact to MinIO successfully.")
        except Exception as e:
            print(f"[ERROR] [Storage] Upload failed: {e}")
            print("-> Check AWS_ACCESS_KEY or Endpoint env vars.")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    print("\n[DONE] CHECK COMPLETE!")
    print("-> Go to http://localhost:5000 to view results.")
    print("-> Go to http://localhost:9001 (Bucket: mlflow-bucket) to view files.")

if __name__ == "__main__":
    run_check()