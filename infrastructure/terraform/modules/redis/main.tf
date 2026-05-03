resource "google_redis_instance" "cache" {
  name           = var.instance_name
  project        = var.project_id
  region         = var.region
  tier           = var.tier
  memory_size_gb = var.memory_size_gb
  redis_version  = var.redis_version

  auth_enabled            = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"

  authorized_network = var.network_id

  dynamic "maintenance_policy" {
    for_each = var.maintenance_policy != null ? [var.maintenance_policy] : []
    content {
      weekly_maintenance_window {
        day = maintenance_policy.value.day
        start_time {
          hours = maintenance_policy.value.start_hour
        }
      }
    }
  }

  redis_configs = {
    maxmemory-policy = "allkeys-lru"
  }

  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }
}
