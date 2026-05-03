resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region

  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = var.network
  subnetwork = var.subnetwork

  # IP allocation for pods and services
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = var.workload_identity_config.workload_pool
  }

  # Network policy
  network_policy {
    enabled = true
  }

  # Maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Monitoring and logging
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }

  # Binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }
}

# Node pools
resource "google_container_node_pool" "pools" {
  for_each = { for pool in var.node_pools : pool.name => pool }

  name       = each.value.name
  cluster    = google_container_cluster.primary.name
  location   = var.region

  initial_node_count = each.value.min_count

  autoscaling {
    min_node_count = each.value.min_count
    max_node_count = each.value.max_count
  }

  management {
    auto_repair  = each.value.auto_repair
    auto_upgrade = each.value.auto_upgrade
  }

  node_config {
    machine_type = each.value.machine_type
    disk_size_gb = each.value.disk_size_gb
    disk_type    = lookup(each.value, "disk_type", "pd-standard")
    preemptible  = each.value.preemptible

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    dynamic "taint" {
      for_each = lookup(each.value, "taints", [])
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}
