module.exports = {
  apps: [
    {
      name: "quant_okx",
      script: ".venv/bin/python",
      args: "-m run.scheduler",
      cwd: "/root/quant_sol_project",
      env: {
        PYTHONPATH: "/root/quant_sol_project",
        PATH: process.env.PATH
      }
    }
  ]
}
