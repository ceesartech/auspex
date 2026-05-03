resource "google_compute_global_address" "private_ip" {
  name          = "${var.instance_name}-private-ip"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = var.network_id
}

resource "google_service_networking_connection" "private_vpc" {
  network                 = var.network_id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip.name]
}

resource "google_sql_database_instance" "primary" {
  name                = var.instance_name
  project             = var.project_id
  region              = var.region
  database_version    = var.database_version
  deletion_protection = var.deletion_protection

  depends_on = [google_service_networking_connection.private_vpc]

  settings {
    tier              = var.tier
    availability_type = var.availability_type
    disk_size         = var.disk_size
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    ip_configuration {
      ipv4_enabled    = false
      private_network = var.network_id
    }

    backup_configuration {
      enabled                        = var.backup_configuration.enabled
      start_time                     = var.backup_configuration.start_time
      point_in_time_recovery_enabled = var.backup_configuration.point_in_time_recovery_enabled
      transaction_log_retention_days = var.backup_configuration.transaction_log_retention_days

      backup_retention_settings {
        retained_backups = var.backup_configuration.retained_backups
      }
    }

    maintenance_window {
      day          = var.maintenance_window.day
      hour         = var.maintenance_window.hour
      update_track = var.maintenance_window.update_track
    }

    dynamic "database_flags" {
      for_each = var.database_flags
      content {
        name  = database_flags.value.name
        value = database_flags.value.value
      }
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }
}

resource "google_sql_database" "database" {
  name     = var.database_name
  project  = var.project_id
  instance = google_sql_database_instance.primary.name
}

resource "google_sql_user" "user" {
  name     = var.database_user
  project  = var.project_id
  instance = google_sql_database_instance.primary.name
  password = var.database_password
}
