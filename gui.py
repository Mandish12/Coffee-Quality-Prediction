import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import Meter
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load and prepare the data
filename = "Coffee_Qlty.csv"
df = pd.read_csv(filename)

cupping_cols = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body",
                "Balance", "Sweetness", "Clean.Cup", "Uniformity"]

df = df.dropna(subset=cupping_cols)
df["QualityScore"] = df[cupping_cols].mean(axis=1)

def classify_quality(score):
    if score >= 8.5:
        return "Excellent"
    elif score >= 7.5:
        return "Good"
    else:
        return "Average"

df["QualityLabel"] = df["QualityScore"].apply(classify_quality)

features = cupping_cols + ["Moisture", "Processing.Method", "Country.of.Origin", "Continent.of.Origin"]
df = df.dropna(subset=features)

X = df[features]
y = df["QualityLabel"]

processing_methods = sorted(df["Processing.Method"].dropna().unique())
countries = sorted(df["Country.of.Origin"].dropna().unique())
continents = sorted(df["Continent.of.Origin"].dropna().unique())

X = pd.get_dummies(X, columns=["Processing.Method", "Country.of.Origin", "Continent.of.Origin"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_scaled, y)

# ------------------ GUI ------------------
class CoffeeApp:
    def __init__(self, master):
        self.master = master
        master.title("☕ Coffee Quality Predictor")
        master.geometry("800x900")
        master.resizable(True, True)

        self.entries = {}

        canvas = tk.Canvas(master)
        scrollbar = ttk.Scrollbar(master, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        title = ttk.Label(self.scrollable_frame, text="Coffee Quality Prediction", font=("Segoe UI", 20, "bold"))
        title.pack(pady=20)

        form_frame = ttk.Frame(self.scrollable_frame, padding=15)
        form_frame.pack(fill="x", padx=30)

        input_section = ttk.LabelFrame(form_frame, text="Cupping Scores", padding=(10, 15))
        input_section.pack(fill="x")

        for col in cupping_cols:
            self._create_slider(input_section, col, 0, 10)

        self._create_slider(input_section, "Moisture", 0, 100)

        # Dropdown Section
        dropdown_section = ttk.LabelFrame(form_frame, text="Attributes", padding=(10, 15))
        dropdown_section.pack(fill="x", pady=(15, 5))

        self.processing_var = tk.StringVar()
        self.country_var = tk.StringVar()
        self.continent_var = tk.StringVar()

        self._create_combobox(dropdown_section, "Processing Method", processing_methods, self.processing_var)
        self._create_combobox(dropdown_section, "Country of Origin", countries, self.country_var)
        self._create_combobox(dropdown_section, "Continent of Origin", continents, self.continent_var)

        ttk.Button(self.scrollable_frame, text="Predict Quality", command=self.predict, bootstyle="success-outline")\
            .pack(pady=25)

        self.result_label = ttk.Label(self.scrollable_frame, text="", font=("Segoe UI", 16, "bold"), anchor="center")
        self.result_label.pack(pady=10)

    def _create_slider(self, parent, label, from_, to):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text=label, width=20).pack(side="left")
        val_label = ttk.Label(frame, text="0.0", width=5)
        val_label.pack(side="right")
        scale = ttk.Scale(frame, from_=from_, to=to, orient="horizontal", length=300,
                          command=lambda val, l=val_label: l.config(text=f"{float(val):.1f}"))
        scale.set((from_ + to) / 2)
        scale.pack(side="left", padx=(10, 0))
        self.entries[label] = scale

    def _create_combobox(self, parent, label_text, options, variable):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text=label_text, width=20).pack(side="left")
        combobox = ttk.Combobox(frame, textvariable=variable, values=options, width=30)
        combobox.pack(side="left")
        combobox.set(options[0])

    def predict(self):
        try:
            input_data = {}
            for col in cupping_cols:
                val = float(self.entries[col].get())
                if not (0 <= val <= 10):
                    raise ValueError(f"{col} must be between 0 and 10.")
                input_data[col] = val

            moisture_val = float(self.entries["Moisture"].get())
            if not (0 <= moisture_val <= 100):
                raise ValueError("Moisture must be between 0 and 100.")
            input_data["Moisture"] = moisture_val

            input_data["Processing.Method"] = self.processing_var.get()
            input_data["Country.of.Origin"] = self.country_var.get()
            input_data["Continent.of.Origin"] = self.continent_var.get()

            df_input = pd.DataFrame([input_data])
            df_input = pd.get_dummies(df_input, columns=["Processing.Method", "Country.of.Origin", "Continent.of.Origin"])

            for col in X.columns:
                if col not in df_input.columns:
                    df_input[col] = 0

            df_input = df_input[X.columns]
            scaled_input = scaler.transform(df_input)

            prediction = clf.predict(scaled_input)[0]
            self.result_label.config(text=f"Predicted Quality: {prediction}", foreground="green")

        except ValueError as ve:
            messagebox.showerror("Invalid Input", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    app = tb.Window(themename="superhero")
    CoffeeApp(app)
    app.mainloop()
