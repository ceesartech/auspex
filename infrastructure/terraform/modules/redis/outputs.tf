output "host" {
  description = "Redis host"
  value       = google_redis_instance.cache.host
}

output "port" {
  description = "Redis port"
  value       = google_redis_instance.cache.port
}

output "auth_string" {
  description = "Redis auth string"
  value       = google_redis_instance.cache.auth_string
  sensitive   = true
}
