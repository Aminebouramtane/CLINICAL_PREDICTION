"""Interface graphique (Tkinter) : saisie d'un patient et diagnostic."""
import tkinter as tk
from tkinter import messagebox, ttk


class ClinicalApp(tk.Tk):
    def __init__(self, predictor, scaler):
        super().__init__()
        self.predictor = predictor
        self.scaler = scaler
        self.title("Diagnostic clinique - virus")
        self.resizable(False, False)
        self.configure(padx=20, pady=20)
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Diagnostic clinique", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 15))

        self.temperature = tk.StringVar(value="37.0")
        self.age = tk.StringVar(value="40")
        self.spo2 = tk.StringVar(value="97")
        self.toux = tk.IntVar()
        self.fatigue = tk.IntVar()
        self.contact = tk.IntVar()

        rows = [
            ("Température (°C)", ttk.Entry(self, textvariable=self.temperature, width=12)),
            ("Âge (ans)", ttk.Entry(self, textvariable=self.age, width=12)),
            ("Saturation O2 (%)", ttk.Entry(self, textvariable=self.spo2, width=12)),
            ("Toux", ttk.Checkbutton(self, variable=self.toux)),
            ("Fatigue", ttk.Checkbutton(self, variable=self.fatigue)),
            ("Contact avec un infecté", ttk.Checkbutton(self, variable=self.contact)),
        ]
        for i, (label, widget) in enumerate(rows, start=1):
            ttk.Label(self, text=label).grid(row=i, column=0, sticky="w", pady=4, padx=(0, 20))
            widget.grid(row=i, column=1, sticky="w")

        ttk.Button(self, text="Diagnostiquer", command=self.on_diagnose).grid(
            row=7, column=0, columnspan=2, pady=15, ipadx=10, ipady=4)

        self.result = tk.Label(self, text="—", font=("Segoe UI", 20, "bold"), width=16, pady=10)
        self.result.grid(row=8, column=0, columnspan=2)
        self.proba = ttk.Label(self, text="")
        self.proba.grid(row=9, column=0, columnspan=2)

    def on_diagnose(self) -> None:
        try:
            # Ordre des colonnes : temperature, toux, fatigue, age, saturation_o2, contact_infecte
            patient = [
                float(self.temperature.get().replace(",", ".")),
                self.toux.get(),
                self.fatigue.get(),
                float(self.age.get()),
                float(self.spo2.get().replace(",", ".")),
                self.contact.get(),
            ]
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez saisir des nombres valides.")
            return

        x = self.scaler.transform(patient)
        verdict = self.predictor.diagnose(x)
        proba = float(self.predictor.model.predict(x)[0])
        color = "#c0392b" if verdict == "Infecté" else "#27ae60"
        self.result.config(text=verdict, fg=color)
        self.proba.config(text=f"Probabilité d'infection : {proba:.1%}")
