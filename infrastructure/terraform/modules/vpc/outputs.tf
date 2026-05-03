output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.vpc.name
}

output "network_id" {
  description = "VPC network ID"
  value       = google_compute_network.vpc.id
}

output "subnets" {
  description = "Map of subnet name to subnet object"
  value       = google_compute_subnetwork.subnets
}
