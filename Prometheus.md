# Metrics collection using Prometheus
--- 
### Following are the steps that can be used to collect cpu and memory metrics from window host laptop

#### 1. Windows Exporter starts 
> .\windows_exporter.exe --collectors.enabled "cpu,mem"    

What happens:
- The application starts a small HTTP server on your machine.
- Default port: 9182
- Only CPU and memory collectors are active (all other metrics are disabled).
- Windows Exporter continuously measures CPU and memory usage of your laptop.

Output:
- Metrics are exposed at http://localhost:9182/metrics in Prometheus format (plain text).      

#### 2. Prometheus scrapes metrics

>scrape_configs:
  - job_name: 'windows'   
    static_configs:
      - targets: ['localhost:9182']


What happens:
- Prometheus reads the prometheus.yml config.
- Every scrape_interval (e.g., 5 seconds), Prometheus makes an HTTP request to http://localhost:9182/metrics.
- Pulls the current CPU and memory metrics.
- This process is called scraping.

#### 3. Prometheus stores metrics

What happens:
- Metrics received from Windows Exporter are stored as time-series data in Prometheus’s internal database.
- Each metric has:
  - Name (e.g., windows_cpu_time_total)
  - Timestamp (when it was collected)
  - Labels (e.g., instance=localhost:9182)

- This data is persisted on disk, so it can be queried later.

#### 4. Query & visualize metrics

What happens:
- We can open Prometheus UI (http://localhost:9090) to query CPU and memory metrics.
- For example:
  - windows_cpu_time_total → total CPU usage over time
  - windows_memory_available_bytes → available memory