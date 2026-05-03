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
    prefix = "staging"
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
  network_name = "betting-system-vpc-staging"
  region       = var.region

  subnets = [
    {
      name          = "gke-subnet"
      ip_cidr_range = "10.20.0.0/20"
      region        = var.region
    },
    {
      name          = "db-subnet"
      ip_cidr_range = "10.20.16.0/24"
      region        = var.region
    }
  ]

  secondary_ip_ranges = {
    gke-subnet = [
      {
        range_name    = "pods"
        ip_cidr_range = "10.24.0.0/14"
      },
      {
        range_name    = "services"
        ip_cidr_range = "10.20.32.0/20"
      }
    ]
  }
}

# GKE Cluster - mid-tier for staging
module "gke" {
  source = "../../modules/gke"

  project_id   = var.project_id
  region       = var.region
  cluster_name = "betting-system-cluster-staging"
  network      = module.vpc.network_name
  subnetwork   = module.vpc.subnets["gke-subnet"].name

  node_pools = [
    {
      name         = "general-pool"
      machine_type = "e2-standard-2"
      min_count    = 1
      max_count    = 3
      disk_size_gb = 50
      disk_type    = "pd-standard"
      preemptible  = true
      auto_repair  = true
      auto_upgrade = true
    }
  ]

  workload_identity_config = {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

# Cloud SQL - mid-tier for staging
module "cloudsql" {
  source = "../../modules/cloudsql"

  project_id       = var.project_id
  region           = var.region
  instance_name    = "betting-system-db-staging"
  database_version = "POSTGRES_15"
  network_id       = module.vpc.network_id

  tier              = "db-custom-2-4096"
  disk_size         = 20
  availability_type = "ZONAL"

  backup_configuration = {
    enabled                        = true
    start_time                     = "03:00"
    point_in_time_recovery_enabled = true
    transaction_log_retention_days = 5
    retained_backups               = 14
  }

  maintenance_window = {
    day          = 7
    hour         = 3
    update_track = "stable"
  }

  deletion_protection = false
  database_password   = var.db_password
}

# Redis - HA for staging
module "redis" {
  source = "../../modules/redis"

  project_id    = var.project_id
  region        = var.region
  environment   = var.environment
  instance_name = "betting-system-redis-staging"
  network_id    = module.vpc.network_id

  tier           = "STANDARD_HA"
  memory_size_gb = 2

  maintenance_policy = {
    day        = "SUNDAY"
    start_hour = 3
  }
}

# GCS
module "gcs" {
  source = "../../modules/gcs"

  project_id  = var.project_id
  environment = var.environment

  buckets = [
    {
      name          = "${var.project_id}-betting-data-staging"
      location      = var.region
      storage_class = "STANDARD"
      versioning    = true
    },
    {
      name          = "${var.project_id}-ml-models-staging"
      location      = var.region
      storage_class = "STANDARD"
      versioning    = true
    },
    {
      name          = "${var.project_id}-backups-staging"
      location      = var.region
      storage_class = "NEARLINE"
    }
  ]
}
