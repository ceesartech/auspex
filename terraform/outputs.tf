output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.betting_cluster.endpoint
  sensitive   = true
}

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.betting_cluster.name
}

output "database_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.betting_db.connection_name
}

output "database_ip" {
  description = "Cloud SQL public IP"
  value       = google_sql_database_instance.betting_db.public_ip_address
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.betting_cache.host
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.betting_cache.port
}

output "data_bucket_url" {
  description = "Data storage bucket URL"
  value       = google_storage_bucket.data_bucket.url
}

output "model_bucket_url" {
  description = "Model artifacts bucket URL"
  value       = google_storage_bucket.model_artifacts.url
}
