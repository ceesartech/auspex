resource "google_storage_bucket" "buckets" {
  for_each = { for b in var.buckets : b.name => b }

  name          = each.value.name
  project       = var.project_id
  location      = each.value.location
  storage_class = each.value.storage_class
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = lookup(each.value, "versioning", false)
  }

  dynamic "lifecycle_rule" {
    for_each = lookup(each.value, "lifecycle_rules", [])
    content {
      action {
        type = lifecycle_rule.value.action.type
      }
      condition {
        age = lifecycle_rule.value.condition.age
      }
    }
  }

  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }
}
