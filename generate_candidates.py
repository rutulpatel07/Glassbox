"""
generate_candidates.py
======================
Generates 1 lakh (100,000) synthetic Indian job candidates.
Saves output to candidates.json using batched writes.

Distributions used:
  - Experience: Exponential (most people have < 10 yrs)
  - Salary: Log-Normal (right-skewed, matches real salary data)
  - GPA: Gaussian (centered at 7.5 / 10)
  - Skills count: Uniform 4-15
"""

import json
import random
import numpy as np
import sys

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Data Pools ────────────────────────────────────────────────────────────────

FIRST_NAMES_M = [
    "Aarav","Arjun","Aditya","Akash","Amit","Ankit","Anirudh","Arnav","Ashish","Ayush",
    "Dev","Dhruv","Gaurav","Harsh","Ishan","Kartik","Kunal","Manish","Mohit","Nikhil",
    "Nitin","Parth","Pranav","Prateek","Rahul","Raj","Rajesh","Rohan","Sagar","Sahil",
    "Saurabh","Shivam","Sidharth","Sourav","Sumit","Tarun","Uday","Varun","Vijay","Vishal",
    "Yash","Abhishek","Anil","Deepak","Girish","Hemant","Kiran","Mayank","Neeraj","Om"
]
FIRST_NAMES_F = [
    "Aanya","Aditi","Akanksha","Ananya","Anjali","Aparna","Bhavna","Deepa","Divya","Esha",
    "Gauri","Isha","Kavya","Kirti","Komal","Kritika","Mahi","Mansi","Meera","Natasha",
    "Neha","Nikita","Nisha","Poonam","Pooja","Priya","Riya","Sakshi","Sana","Shreya",
    "Simran","Sneha","Sonali","Swati","Tanvi","Trisha","Uma","Vandana","Varsha","Vidya",
    "Pallavi","Rashmi","Ruhi","Shikha","Shalini","Tanya","Urvashi","Veena","Yamini","Zara"
]
LAST_NAMES = [
    "Sharma","Verma","Gupta","Singh","Kumar","Patel","Shah","Mehta","Joshi","Rao",
    "Nair","Pillai","Iyer","Krishnan","Reddy","Naidu","Chandra","Mishra","Tiwari","Pandey",
    "Bose","Das","Sen","Ghosh","Mukherjee","Banerjee","Chatterjee","Jain","Agarwal","Kapoor",
    "Malhotra","Khanna","Bhatia","Arora","Chopra","Sinha","Trivedi","Dubey","Yadav","Thakur",
    "Desai","Parikh","Bhatt","Kulkarni","Patil","Shinde","More","Kaur","Gill","Dhaliwal"
]

# Skills by domain
SKILLS = {
    "tech": [
        "Python","Java","JavaScript","TypeScript","Go","Rust","C++","C#","PHP","Ruby","Scala",
        "React","Vue.js","Angular","Node.js","Django","Flask","FastAPI","Spring Boot","Laravel",
        "TensorFlow","PyTorch","scikit-learn","Keras","OpenCV","NLTK","spaCy","Hugging Face",
        "SQL","PostgreSQL","MySQL","MongoDB","Redis","Elasticsearch","Cassandra","DynamoDB",
        "AWS","GCP","Azure","Docker","Kubernetes","Terraform","Jenkins","GitHub Actions",
        "Machine Learning","Deep Learning","NLP","Computer Vision","Data Science","Statistics",
        "Pandas","NumPy","Apache Spark","Hadoop","Kafka","Airflow","dbt","Snowflake",
        "Git","Linux","REST API","GraphQL","Microservices","System Design","LeetCode"
    ],
    "management": [
        "Project Management","Agile","Scrum","Kanban","JIRA","Confluence","Asana",
        "Team Leadership","Stakeholder Management","Budget Planning","Risk Management",
        "Product Roadmap","OKR","Strategic Planning","Change Management","PnL Management",
        "Excel","PowerPoint","Tableau","Power BI","Data Analysis","Business Analysis"
    ],
    "design": [
        "Figma","Sketch","Adobe XD","Photoshop","Illustrator","InDesign","After Effects",
        "UI Design","UX Design","Wireframing","Prototyping","User Research","Usability Testing",
        "Design Systems","Typography","Color Theory","Motion Design","Responsive Design"
    ],
    "finance": [
        "Financial Analysis","Excel","Power BI","Tableau","SQL","Accounting","Valuation",
        "Risk Assessment","Portfolio Management","Bloomberg Terminal","SAP","QuickBooks",
        "GAAP","IFRS","DCF Modeling","Python","R","Monte Carlo Simulation","VBA","FRM"
    ],
    "marketing": [
        "SEO","SEM","Google Analytics","Meta Ads","Content Marketing","Email Marketing",
        "CRM","HubSpot","Salesforce","Brand Strategy","Market Research","A/B Testing",
        "Conversion Optimization","Social Media","Copywriting","Video Marketing","Growth Hacking"
    ]
}

EDUCATION_DEGREES = [
    "B.Tech","M.Tech","MBA","B.Sc","M.Sc","BCA","MCA","B.E","Ph.D","Diploma","B.Com","M.Com"
]
DEGREE_WEIGHTS_GEN = [0.32,0.14,0.14,0.08,0.08,0.05,0.05,0.04,0.02,0.03,0.02,0.03]

EDUCATION_FIELDS = {
    "tech": [
        "Computer Science","Information Technology","Electronics & Communication",
        "Electrical Engineering","Data Science","AI & Machine Learning","Software Engineering"
    ],
    "management": [
        "Business Administration","Finance","Marketing","Operations Management","Human Resources"
    ],
    "other": ["Mathematics","Statistics","Physics","Economics","Commerce"]
}

INSTITUTIONS = {
    "tier1": [
        "IIT Bombay","IIT Delhi","IIT Madras","IIT Kanpur","IIT Kharagpur",
        "IIM Ahmedabad","IIM Bangalore","IIM Calcutta","BITS Pilani","NIT Trichy",
        "IIT Roorkee","IIT Hyderabad","IISc Bangalore"
    ],
    "tier2": [
        "VIT Vellore","SRM University","Manipal University","NMIMS Mumbai","Symbiosis Pune",
        "PSG Tech","VJTI Mumbai","COEP Pune","DAIICT","Thapar University",
        "NIT Surat","NIT Warangal","DTU Delhi","PEC Chandigarh"
    ],
    "tier3": [
        "Gujarat Technological University","Mumbai University","Pune University",
        "Anna University","Rajasthan Technical University","Osmania University",
        "Bangalore University","Calcutta University","Various State Engineering Colleges"
    ]
}

COMPANIES = {
    "tier1": [
        "Google","Microsoft","Amazon","Meta","Apple","Netflix","LinkedIn","Stripe",
        "Goldman Sachs","Morgan Stanley","McKinsey","BCG","Deloitte",
        "TCS","Infosys","Wipro","HCL","Accenture","IBM"
    ],
    "tier2": [
        "Flipkart","Ola","Swiggy","Zomato","Razorpay","CRED","Zepto","Meesho",
        "Paytm","PhonePe","Dream11","ShareChat","Freshworks","Zoho","Intuit India",
        "Byju's","Unacademy","Naukri","MakeMyTrip","Myntra","Bigbasket"
    ],
    "tier3": [
        "Tech Mahindra","Mphasis","Hexaware","Mindtree","L&T Infotech",
        "Persistent Systems","Mastek","Cyient","NIIT Technologies",
        "Various Startups","Mid-size IT Firms","BPO Companies","Local Firms"
    ]
}

ROLES = {
    "tech": [
        "Software Engineer","Senior Software Engineer","Staff Engineer","Principal Engineer",
        "Data Scientist","ML Engineer","DevOps Engineer","Full Stack Developer",
        "Backend Developer","Frontend Developer","Mobile Developer","QA Engineer",
        "Security Engineer","Database Administrator","Cloud Architect","Tech Lead"
    ],
    "management": [
        "Project Manager","Product Manager","Engineering Manager","Program Manager",
        "Scrum Master","Operations Manager","Business Analyst","Strategy Consultant"
    ],
    "design": ["UI Designer","UX Designer","Product Designer","Visual Designer","Motion Designer"],
    "finance": ["Financial Analyst","Data Analyst","Risk Analyst","Investment Analyst","Quant Analyst"],
    "marketing": ["Digital Marketer","SEO Specialist","Content Manager","Growth Manager","Brand Manager"]
}

CERTIFICATIONS = {
    "tech": [
        "AWS Solutions Architect Associate","AWS Solutions Architect Professional",
        "GCP Professional Data Engineer","GCP Professional Cloud Architect",
        "Azure Administrator Associate","Kubernetes CKA","Terraform Associate",
        "Red Hat RHCE","Google Data Engineer","Databricks Certified",
        "TensorFlow Developer Certificate","Oracle Java SE Certified",
        "MongoDB Certified Developer","Cisco CCNA","CompTIA Security+"
    ],
    "management": [
        "PMP","PRINCE2 Practitioner","Six Sigma Green Belt","Six Sigma Black Belt",
        "Certified Scrum Master","SAFe Agilist","ITIL V4","Prince2 Foundation"
    ],
    "design": [
        "Google UX Design Professional","Adobe Certified Expert",
        "Interaction Design Foundation","Nielsen Norman UX Certified"
    ],
    "finance": ["CFA Level 1","CFA Level 2","CFA Level 3","FRM Part 1","FRM Part 2","CA","CPA","CMA"],
    "marketing": [
        "Google Analytics Certified","Google Ads Certified","HubSpot Marketing Certified",
        "Facebook Blueprint Certified","Hootsuite Social Marketing"
    ]
}

LOCATIONS = [
    "Bangalore","Mumbai","Delhi","Hyderabad","Pune","Chennai","Kolkata",
    "Ahmedabad","Surat","Jaipur","Lucknow","Chandigarh","Coimbatore",
    "Noida","Gurgaon","Kochi","Indore","Nagpur","Bhopal","Vadodara",
    "Bhubaneswar","Visakhapatnam","Patna","Ranchi","Raipur"
]

# ── Generator ─────────────────────────────────────────────────────────────────

def _random_name():
    gender = random.choice(["M", "F"])
    fname = random.choice(FIRST_NAMES_M if gender == "M" else FIRST_NAMES_F)
    lname = random.choice(LAST_NAMES)
    return f"{fname} {lname}", gender

def _generate_email(name: str, uid: int) -> str:
    parts = name.lower().split()
    domains = ["gmail.com","yahoo.com","outlook.com","hotmail.com","protonmail.com"]
    return f"{parts[0]}.{parts[-1]}{uid % 999}@{random.choice(domains)}"

def _generate_work_history(experience_years: int, domain: str):
    if experience_years == 0:
        return []
    history = []
    remaining = experience_years
    while remaining > 0:
        stint = min(random.randint(1, 5), remaining)
        tier = random.choices(["tier1","tier2","tier3"], weights=[0.12, 0.33, 0.55])[0]
        history.append({
            "company": random.choice(COMPANIES[tier]),
            "company_tier": tier,
            "role": random.choice(ROLES.get(domain, ROLES["tech"])),
            "years": stint
        })
        remaining -= stint
    return history

def generate_candidate(uid: int, domain: str) -> dict:
    name, gender = _random_name()

    # Experience: exponential, capped at 30
    exp = int(np.random.exponential(5.5))
    exp = min(exp, 30)

    # Skills
    pool = SKILLS.get(domain, SKILLS["tech"])
    n_skills = random.randint(4, min(15, len(pool)))
    skills = random.sample(pool, n_skills)

    # Education
    degree = random.choices(EDUCATION_DEGREES, weights=DEGREE_WEIGHTS_GEN)[0]
    field_cat = random.choices(["tech","management","other"], weights=[0.50, 0.30, 0.20])[0]
    field = random.choice(EDUCATION_FIELDS[field_cat])
    inst_tier = random.choices(["tier1","tier2","tier3"], weights=[0.05, 0.25, 0.70])[0]
    institution = random.choice(INSTITUTIONS[inst_tier])
    gpa = round(float(np.clip(np.random.normal(7.5, 0.85), 4.5, 10.0)), 2)

    # Certifications
    n_certs = random.choices([0,1,2,3,4], weights=[0.40, 0.30, 0.15, 0.10, 0.05])[0]
    cert_pool = CERTIFICATIONS.get(domain, CERTIFICATIONS["tech"])
    certs = random.sample(cert_pool, min(n_certs, len(cert_pool)))

    # Salary: log-normal around experience-based anchor
    anchor = 300_000 + exp * 130_000
    salary = int(np.clip(np.random.lognormal(np.log(anchor), 0.35), 150_000, 8_000_000))

    return {
        "id": f"C{uid:07d}",
        "name": name,
        "gender": gender,
        "email": _generate_email(name, uid),
        "phone": f"+91 {random.randint(7000000000, 9999999999)}",
        "location": random.choice(LOCATIONS),
        "willing_to_relocate": random.random() > 0.38,
        "skills": skills,
        "experience_years": exp,
        "work_history": _generate_work_history(exp, domain),
        "education": {
            "degree": degree,
            "field": field,
            "institution": institution,
            "institution_tier": inst_tier,
            "gpa": gpa
        },
        "certifications": certs,
        "salary_expectation": salary,
        "notice_period_days": random.choice([0, 15, 30, 30, 60, 60, 90]),
        "domain": domain
    }


def generate_dataset(n: int = 100_000, output_path: str = "candidates.json",
                     batch_size: int = 10_000) -> None:
    """
    Generate N candidates and stream-write to JSON.
    Uses batching so memory stays bounded even for 1L records.
    """
    domains = ["tech","management","design","finance","marketing"]
    d_weights = [0.50, 0.20, 0.10, 0.10, 0.10]

    print(f"Generating {n:,} candidates  →  {output_path}")

    with open(output_path, "w") as fout:
        fout.write("[\n")
        for i in range(n):
            domain = random.choices(domains, weights=d_weights)[0]
            candidate = generate_candidate(i + 1, domain)
            suffix = ",\n" if i < n - 1 else "\n"
            fout.write(json.dumps(candidate) + suffix)
            if (i + 1) % batch_size == 0:
                pct = (i + 1) / n * 100
                print(f"  {i+1:>8,} / {n:,}  ({pct:.0f}%)")
                sys.stdout.flush()
        fout.write("]\n")

    print(f"✓ Done → {output_path}  ({n:,} candidates)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",     type=int, default=100_000, help="Number of candidates")
    ap.add_argument("--out",   default="candidates.json",  help="Output JSON path")
    ap.add_argument("--batch", type=int, default=10_000,   help="Batch size for progress")
    args = ap.parse_args()
    generate_dataset(args.n, args.out, args.batch)
