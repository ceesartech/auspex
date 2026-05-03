terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "betting-system-terraform-state"
    prefix = "prod"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
module "vpc" {
  source = "../../modules/vpc"

  project_id   = var.project_id
  network_name = "betting-system-vpc"
  region       = var.region

  subnets = [
    {
      name          = "gke-subnet"
      ip_cidr_range = "10.0.0.0/20"
      region        = var.region
    },
    {
      name          = "db-subnet"
      ip_cidr_range = "10.0.16.0/24"
      region        = var.region
    }
  ]

  secondary_ip_ranges = {
    gke-subnet = [
      {
        range_name    = "pods"
        ip_cidr_range = "10.4.0.0/14"
      },
      {
        range_name    = "services"
        ip_cidr_range = "10.0.32.0/20"
      }
    ]
  }
}

# GKE Cluster
module "gke" {
  source = "../../modules/gke"

  project_id   = var.project_id
  region       = var.region
  cluster_name = "betting-system-cluster"
  network      = module.vpc.network_name
  subnetwork   = module.vpc.subnets["gke-subnet"].name

  # Node pool configuration
  node_pools = [
    {
      name         = "general-pool"
      machine_type = "e2-medium"
      min_count    = 1
      max_count    = 5
      disk_size_gb = 50
      disk_type    = "pd-standard"
      preemptible  = true
      auto_repair  = true
      auto_upgrade = true
    },
    {
      name         = "ml-pool"
      machine_type = "n1-standard-4"
      min_count    = 0
      max_count    = 2
      disk_size_gb = 100
      preemptible  = true
      auto_repair  = true
      auto_upgrade = true
      taints = [
        {
          key    = "workload"
          value  = "ml"
          effect = "NoSchedule"
        }
      ]
    }
  ]

  # Enable Workload Identity
  workload_identity_config = {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

# Cloud SQL PostgreSQL
module "cloudsql" {
  source = "../../modules/cloudsql"

  project_id       = var.project_id
  region           = var.region
  instance_name    = "betting-system-db"
  database_version = "POSTGRES_15"
  network_id       = module.vpc.network_id

  tier = "db-custom-2-4096"

  database_flags = [
    {
      name  = "max_connections"
      value = "200"
    },
    {
      name  = "shared_buffers"
      value = "1024000"
    }
  ]

  backup_configuration = {
    enabled                        = true
    start_time                     = "03:00"
    point_in_time_recovery_enabled = true
    transaction_log_retention_days = 7
    retained_backups               = 30
  }

  availability_type = "REGIONAL"

  maintenance_window = {
    day          = 7
    hour         = 3
    update_track = "stable"
  }

  deletion_protection = true
  database_password   = var.db_password
}

# Cloud Memorystore (Redis)
module "redis" {
  source = "../../modules/redis"

  project_id    = var.project_id
  region        = var.region
  environment   = var.environment
  instance_name = "betting-system-redis"
  network_id    = module.vpc.network_id

  tier           = "STANDARD_HA"
  memory_size_gb = 2
  redis_version  = "REDIS_7_0"

  maintenance_policy = {
    day        = "SUNDAY"
    start_hour = 3
    duration   = "4h"
  }
}

# Google Cloud Storage
module "gcs" {
  source = "../../modules/gcs"

  project_id  = var.project_id
  environment = var.environment

  buckets = [
    {
      name          = "${var.project_id}-betting-data"
      location      = var.region
      storage_class = "STANDARD"
      versioning    = true
      lifecycle_rules = [
        {
          action = {
            type = "Delete"
          }
          condition = {
            age = 90
          }
        }
      ]
    },
    {
      name          = "${var.project_id}-ml-models"
      location      = var.region
      storage_class = "STANDARD"
      versioning    = true
    },
    {
      name          = "${var.project_id}-backups"
      location      = var.region
      storage_class = "NEARLINE"
    }
  ]
}

# Cloud Scheduler (for automated tasks)
resource "google_cloud_scheduler_job" "model_retraining" {
  name        = "model-retraining-weekly"
  description = "Trigger weekly model retraining"
  schedule    = "0 2 * * 0"
  time_zone   = "America/Denver"

  http_target {
    http_method = "POST"
    uri         = "https://api.betting-system.com/api/v1/models/retrain"

    headers = {
      "Content-Type" = "application/json"
    }

    oidc_token {
      service_account_email = google_service_account.api.email
    }
  }
}

# Service Account for API
resource "google_service_account" "api" {
  account_id   = "betting-system-api"
  display_name = "Betting System API Service Account"
}

# IAM bindings
resource "google_project_iam_member" "api_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "api_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.api.email}"
}
