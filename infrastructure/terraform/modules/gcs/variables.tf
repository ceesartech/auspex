variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "buckets" {
  description = "List of GCS bucket configurations"
  type = list(object({
    name          = string
    location      = string
    storage_class = string
    versioning    = optional(bool, false)
    lifecycle_rules = optional(list(object({
      action = object({
        type = string
      })
      condition = object({
        age = number
      })
    })), [])
  }))
}
