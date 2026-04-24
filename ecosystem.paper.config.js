const path = require("path");

const projectRoot = __dirname;

module.exports = {
  apps: [
    {
      name: "quant_okx_paper",
      cwd: projectRoot,
      script: path.join(projectRoot, ".venv/bin/python"),
      args: "-m run.live_trading_monitor",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        PYTHONPATH: projectRoot,
        PATH: process.env.PATH,
        TELEGRAM_ENABLED: process.env.TELEGRAM_ENABLED || "0"
      }
    },
    {
      name: "quant_okx_dashboard",
      cwd: projectRoot,
      script: path.join(projectRoot, ".venv/bin/python"),
      args: "-m run.dashboard_server",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        PYTHONPATH: projectRoot,
        PATH: process.env.PATH,
        DASHBOARD_HOST: process.env.DASHBOARD_HOST || "0.0.0.0",
        DASHBOARD_PORT: process.env.DASHBOARD_PORT || "8787"
      }
    }
  ]
};
