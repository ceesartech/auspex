variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "instance_name" {
  description = "Cloud SQL instance name"
  type        = string
}

variable "database_version" {
  description = "Database version"
  type        = string
  default     = "POSTGRES_15"
}

variable "tier" {
  description = "Machine tier"
  type        = string
}

variable "disk_size" {
  description = "Disk size in GB"
  type        = number
  default     = 20
}

variable "availability_type" {
  description = "Availability type (ZONAL or REGIONAL)"
  type        = string
  default     = "ZONAL"
}

variable "network_id" {
  description = "VPC network ID for private IP"
  type        = string
}

variable "database_name" {
  description = "Database name"
  type        = string
  default     = "betting_system"
}

variable "database_user" {
  description = "Database user"
  type        = string
  default     = "betting_user"
}

variable "database_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = true
}

variable "backup_configuration" {
  description = "Backup configuration"
  type = object({
    enabled                        = bool
    start_time                     = string
    point_in_time_recovery_enabled = bool
    transaction_log_retention_days = number
    retained_backups               = number
  })
  default = {
    enabled                        = true
    start_time                     = "03:00"
    point_in_time_recovery_enabled = true
    transaction_log_retention_days = 7
    retained_backups               = 7
  }
}

variable "maintenance_window" {
  description = "Maintenance window configuration"
  type = object({
    day          = number
    hour         = number
    update_track = string
  })
  default = {
    day          = 7
    hour         = 3
    update_track = "stable"
  }
}

variable "database_flags" {
  description = "Database flags"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}
