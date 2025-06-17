module.exports = {
  apps: [
    {
      name: "quant_okx",
      script: ".venv/bin/python",
      args: "run/scheduler.py",
      cwd: "/root/quant_sol_project",
      env: {
        PYTHONPATH: "/root/quant_sol_project",   // ✅加PYTHONPATH
        PATH: process.env.PATH,                  // ✅继承 PATH
      }
    }
  ]
}
