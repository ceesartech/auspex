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
    prefix = "dev"
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
  network_name = "betting-system-vpc-dev"
  region       = var.region

  subnets = [
    {
      name          = "gke-subnet"
      ip_cidr_range = "10.10.0.0/20"
      region        = var.region
    },
    {
      name          = "db-subnet"
      ip_cidr_range = "10.10.16.0/24"
      region        = var.region
    }
  ]

  secondary_ip_ranges = {
    gke-subnet = [
      {
        range_name    = "pods"
        ip_cidr_range = "10.14.0.0/14"
      },
      {
        range_name    = "services"
        ip_cidr_range = "10.10.32.0/20"
      }
    ]
  }
}

# GKE Cluster - minimal for dev
module "gke" {
  source = "../../modules/gke"

  project_id   = var.project_id
  region       = var.region
  cluster_name = "betting-system-cluster-dev"
  network      = module.vpc.network_name
  subnetwork   = module.vpc.subnets["gke-subnet"].name

  node_pools = [
    {
      name         = "general-pool"
      machine_type = "e2-medium"
      min_count    = 1
      max_count    = 2
      disk_size_gb = 30
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

# Cloud SQL - minimal for dev
module "cloudsql" {
  source = "../../modules/cloudsql"

  project_id       = var.project_id
  region           = var.region
  instance_name    = "betting-system-db-dev"
  database_version = "POSTGRES_15"
  network_id       = module.vpc.network_id

  tier              = "db-f1-micro"
  disk_size         = 10
  availability_type = "ZONAL"

  backup_configuration = {
    enabled                        = true
    start_time                     = "03:00"
    point_in_time_recovery_enabled = false
    transaction_log_retention_days = 3
    retained_backups               = 7
  }

  maintenance_window = {
    day          = 7
    hour         = 3
    update_track = "stable"
  }

  deletion_protection = false
  database_password   = var.db_password
}

# Redis - minimal for dev
module "redis" {
  source = "../../modules/redis"

  project_id    = var.project_id
  region        = var.region
  environment   = var.environment
  instance_name = "betting-system-redis-dev"
  network_id    = module.vpc.network_id

  tier           = "BASIC"
  memory_size_gb = 1
}

# GCS
module "gcs" {
  source = "../../modules/gcs"

  project_id  = var.project_id
  environment = var.environment

  buckets = [
    {
      name          = "${var.project_id}-betting-data-dev"
      location      = var.region
      storage_class = "STANDARD"
    },
    {
      name          = "${var.project_id}-ml-models-dev"
      location      = var.region
      storage_class = "STANDARD"
      versioning    = true
    }
  ]
}
