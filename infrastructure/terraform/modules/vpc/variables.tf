variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "network_name" {
  description = "VPC network name"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "subnets" {
  description = "List of subnets to create"
  type = list(object({
    name          = string
    ip_cidr_range = string
    region        = string
  }))
}

variable "secondary_ip_ranges" {
  description = "Secondary IP ranges for subnets (keyed by subnet name)"
  type = map(list(object({
    range_name    = string
    ip_cidr_range = string
  })))
  default = {}
}
