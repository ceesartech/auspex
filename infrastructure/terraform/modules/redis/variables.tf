variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "instance_name" {
  description = "Redis instance name"
  type        = string
}

variable "tier" {
  description = "Redis tier (BASIC or STANDARD_HA)"
  type        = string
  default     = "BASIC"
}

variable "memory_size_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 1
}

variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "REDIS_7_0"
}

variable "network_id" {
  description = "VPC network ID"
  type        = string
}

variable "maintenance_policy" {
  description = "Maintenance policy configuration"
  type = object({
    day        = string
    start_hour = number
    duration   = optional(string, "4h")
  })
  default = null
}
