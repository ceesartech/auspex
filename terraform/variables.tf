variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "betting-system"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "database_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "betting_system"
}

variable "database_user" {
  description = "PostgreSQL database user"
  type        = string
  default     = "betting_user"
}

variable "database_password" {
  description = "PostgreSQL database password"
  type        = string
  sensitive   = true
}

variable "api_url" {
  description = "Base URL of the API service"
  type        = string
  default     = "https://api.betting-system.com"
}
