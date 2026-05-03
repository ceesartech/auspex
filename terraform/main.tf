terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "betting-system-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ============= GKE CLUSTER =============

resource "google_container_cluster" "betting_cluster" {
  name     = "${var.project_name}-cluster"
  location = var.zone

  # Free tier: 1 zonal cluster is free
  initial_node_count = 1

  node_config {
    machine_type = "e2-medium" # Free tier eligible
    disk_size_gb = 30
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/trace.append",
    ]

    labels = {
      app         = var.project_name
      environment = var.environment
    }

    tags = ["betting-system"]
  }

  # Enable basic monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  # Network policy
  network_policy {
    enabled = true
  }

  # Maintenance window (off-peak)
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }
}

# ============= CLOUD SQL (PostgreSQL) =============

resource "google_sql_database_instance" "betting_db" {
  name             = "${var.project_name}-db"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = "db-f1-micro" # Free tier
    availability_type = "ZONAL"
    disk_size         = 10
    disk_type         = "PD_HDD"

    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "all"
        value = "0.0.0.0/0"
      }
    }

    backup_configuration {
      enabled    = true
      start_time = "04:00"
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "betting_database" {
  name     = var.database_name
  instance = google_sql_database_instance.betting_db.name
}

resource "google_sql_user" "betting_user" {
  name     = var.database_user
  instance = google_sql_database_instance.betting_db.name
  password = var.database_password
}

# ============= CLOUD STORAGE =============

resource "google_storage_bucket" "data_bucket" {
  name          = "${var.project_id}-betting-data"
  location      = var.region
  force_destroy = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 365
    }
  }

  uniform_bucket_level_access = true
}

resource "google_storage_bucket" "model_artifacts" {
  name          = "${var.project_id}-model-artifacts"
  location      = var.region
  force_destroy = false

  versioning {
    enabled = true
  }

  uniform_bucket_level_access = true
}

# ============= MEMORYSTORE (Redis) =============

resource "google_redis_instance" "betting_cache" {
  name           = "${var.project_name}-cache"
  tier           = "BASIC" # Free tier
  memory_size_gb = 1
  region         = var.region

  redis_version = "REDIS_7_0"

  labels = {
    app         = var.project_name
    environment = var.environment
  }
}

# ============= CLOUD SCHEDULER (for cron jobs) =============

resource "google_cloud_scheduler_job" "daily_scraping" {
  name        = "${var.project_name}-daily-scraping"
  description = "Trigger daily stats scraping"
  schedule    = "0 6 * * *"
  time_zone   = "America/Denver"

  http_target {
    http_method = "POST"
    uri         = "${var.api_url}/api/v1/scraping/trigger/daily"

    headers = {
      "Content-Type" = "application/json"
    }
  }
}

resource "google_cloud_scheduler_job" "odds_scraping" {
  name        = "${var.project_name}-odds-scraping"
  description = "Trigger minute-by-minute odds scraping"
  schedule    = "* * * * *"
  time_zone   = "America/Denver"

  http_target {
    http_method = "POST"
    uri         = "${var.api_url}/api/v1/scraping/trigger/odds"

    headers = {
      "Content-Type" = "application/json"
    }
  }
}
