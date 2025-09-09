from fpdf import FPDF

# Türkçe karakterleri İngilizce karaktere çeviren fonksiyon
def turkce_karakterleri_cevir(text):
    mapping = {
        'İ':'I', 'ı':'i', 'Ğ':'G', 'ğ':'g', 'Ş':'S', 'ş':'s',
        'Ü':'U', 'ü':'u', 'Ö':'O', 'ö':'o', 'Ç':'C', 'ç':'c'
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

pdf = FPDF()
pdf.add_page()

# Başlık
pdf.set_font("Arial", "B", 16)
pdf.multi_cell(0, 10, turkce_karakterleri_cevir("Havelsan Web Sitesi Icerik Ozeti"), align='C')
pdf.ln(10)

pdf.set_font("Arial", "", 12)

# İçerik bilgileri
content = {
    "Kurumsal": {
        "Hakkımızda": "1982 yılında kurulan Havelsan, Türk Silahlı Kuvvetleri ve kamu için yüksek teknoloji ürün ve hizmetler sunmaktadır.",
        "Yönetim Kurulu": "Havelsan'ın yönetim kurulu, stratejik kararları alır ve şirketin hedeflerini belirler.",
        "Üst Yönetim": "Üst yönetim, günlük operasyonları yönetir ve şirketin vizyonunu uygular.",
        "İş Ekosistemi": "Havelsan, çeşitli kamu ve özel sektör iş ortaklarıyla iş birliği yapmaktadır.",
        "Stratejik Plan": "Uzun vadeli hedefler, Ar-Ge yatırımları ve büyüme planları bu başlık altında yer alır.",
        "Sürdürülebilirlik": "Çevresel, sosyal ve ekonomik sürdürülebilirlik çalışmaları ve raporları paylaşılır.",
        "Dünyada Biz": "Havelsan'ın uluslararası faaliyetleri ve iş birlikleri hakkında bilgiler.",
        "Sosyal Sorumluluk Projeleri": "Eğitim, teknoloji ve topluma katkı projeleri anlatılır.",
        "Politikalar": "Kurumsal yönetim ve etik politikalar."
    },
    "Faaliyet Alanlarımız": {
        "Komuta Kontrol ve Savunma": "Askeri ve sivil uygulamalara yönelik komuta kontrol çözümleri.",
        "Askeri Simülasyon ve Eğitim": "Simülasyon sistemleri ve eğitim platformları.",
        "Sivil Havacılık": "Sivil havacılık yazılımları ve yönetim sistemleri.",
        "Bilgi ve İletişim Teknolojileri": "Bilişim çözümleri, IoT ve güvenlik uygulamaları.",
        "İnsansız Otonom Sistemler": "Dronlar, insansız deniz ve kara araçları.",
        "Siber Güvenlik": "Kritik sistemler için siber güvenlik çözümleri."
    },
    "Çözümler": {
        "Komuta Kontrol ve Savunma": "Askeri ve savunma sistemlerinin entegre yönetimi için çözümler.",
        "Simülasyon ve Eğitim": "Eğitim simülasyonları ve sanal ortam sistemleri.",
        "Bilgi ve İletişim Teknolojileri": "Kurumsal yazılım, veri analitiği ve güvenlik çözümleri.",
        "Entegre Güvenlik": "Fiziksel ve dijital güvenlik çözümleri.",
        "AR / MR / VR": "Artırılmış ve sanal gerçeklik tabanlı uygulamalar."
    },
    "Hizmetler": {
        "Savaş Yönetim Sistemleri Entegrasyon Hizmetleri": "Komuta kontrol sistemleri entegrasyonu.",
        "Entegre Lojistik Destek": "Sistemlerin bakım ve lojistik desteği.",
        "Eğitim Hizmetleri": "Simülasyon ve saha eğitimleri.",
        "Siber Güvenlik": "Kritik altyapılar için güvenlik hizmetleri."
    },
    "İnovasyon": {
        "Yaklaşımımız": "Yenilikçi teknolojiler geliştirme ve Ar-Ge odaklı yaklaşım.",
        "Yeni Teknolojiler": "Yapay zekâ, otonom sistemler ve yazılım çözümleri.",
        "Teknoloji Yönetimi": "Projelerde teknoloji yönetimi ve süreç optimizasyonu.",
        "İnovasyon Yönetimi": "İnovatif fikirlerin uygulanması ve geliştirilmesi.",
        "Teşvik Yönetimi": "Ar-Ge teşvikleri ve destek programları.",
        "Fikri ve Sınai Mülkiyet Hakları": "Patent, marka ve tasarım süreçleri.",
        "İnovasyon Programları": "İç ve dış inovasyon programları ve iş birlikleri."
    },
    "Medya": {
        "Haberler ve Basın Bültenleri": "Güncel haberler ve basın açıklamaları.",
        "HAVELSAN Dergi": "Şirket dergisi ve yayınlar.",
        "Dijital Bülten": "Online bülten ve güncel bilgiler.",
        "Ar-Ge, Teknoloji ve İnovasyon Bülteni": "Ar-Ge faaliyetleri hakkında bilgiler.",
        "Siber Güvenlik Bülteni": "Siber güvenlik ile ilgili gelişmeler.",
        "Kurumsal Kimlik": "Marka ve kurumsal tanıtım materyalleri."
    },
    "Kariyer": {
        "HAVELSANLI OLMAK": "Çalışan deneyimleri ve şirket kültürü.",
        "Genel Müdürümüzün Mesajı": "Genel müdürün vizyon ve mesajları.",
        "HAVELSAN Teknoloji Kampüsü": "Teknoloji geliştirme merkezi ve imkanları.",
        "Eğitim ve Gelişim": "Çalışan eğitim ve kariyer gelişim programları.",
        "Kariyer Fuarları": "Etkinlikler ve staj imkanları.",
        "Teknoloji Kampları": "Yaz kampları ve teknoloji odaklı etkinlikler.",
        "Teknik Gezi": "Şirket ve proje tanıtım gezileri.",
        "Teknofest": "Teknofest etkinlikleri ve yarışmalar.",
        "İşe Alım Süreci": "Kariyer başvuru ve işe alım süreçleri.",
        "YETENEK PROGRAMLARI": "Staj ve genç yetenek programları."
    },
    "Bize Ulaşın": {
        "Adres ve Telefonlar": "Merkez ofis adresi ve iletişim numaraları.",
        "Bize Yazın": "İletişim formu ile mesaj gönderme."
    }
}

# PDF yazma
for section, items in content.items():
    pdf.set_font("Arial", "B", 14)
    pdf.multi_cell(0, 8, turkce_karakterleri_cevir(section))
    pdf.set_font("Arial", "", 12)
    for item_title, item_content in items.items():
        pdf.multi_cell(0, 6, "- " + turkce_karakterleri_cevir(item_title) + ": " + turkce_karakterleri_cevir(item_content))
    pdf.ln(4)

# PDF kaydetme
file_path = "havelsan.pdf"
pdf.output(file_path)
print(f"PDF oluşturuldu: {file_path}")
