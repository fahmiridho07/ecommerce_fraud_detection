const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, HeadingLevel, AlignmentType,
        Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, LevelFormat } = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "outputs", "figures");
const OUT = path.join(ROOT, "docs", "KAJIAN_PENYEBAB_AE_v2.docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "AAB2BD" };
const borders = { top: border, bottom: border, left: border, right: border };
const CW = 9360;

function img(file, w, h) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)),
      transformation: { width: w, height: h },
      altText: { title: file, description: file, name: file } })],
  });
}
function h1(t){return new Paragraph({heading:HeadingLevel.HEADING_1,children:[new TextRun(t)]});}
function h2(t){return new Paragraph({heading:HeadingLevel.HEADING_2,children:[new TextRun(t)]});}
function p(t,opts={}){return new Paragraph({spacing:{after:120},children:[new TextRun({text:t,...opts})]});}
function bullet(t){return new Paragraph({numbering:{reference:"b",level:0},children:[new TextRun(t)]});}
function cap(t){return new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:200},children:[new TextRun({text:t,italics:true,size:18,color:"555555"})]});}

const masterRows = [
  ["ae_mlp_stack","Skor neural AE→MLP (OOF)","0,82174","−0,00118","0,807","seri","tie"],
  ["iforest_latent","IsolationForest pd laten AE","0,81851","−0,00441","1,000","lebih buruk","w"],
  ["vae_anomaly","Variational AE (anomali)","0,81804","−0,00487","1,000","lebih buruk","w"],
  ["recon_error","Error rekonstruksi (semua V)","0,81768","−0,00524","1,000","lebih buruk","w"],
  ["one_class_anomaly","Error AE one-class (normal)","0,81690","−0,00602","1,000","lebih buruk","w"],
  ["contrast_anomaly","Error normal-AE vs fraud-AE","0,81667","−0,00624","1,000","lebih buruk","w"],
  ["latent_distance","Jarak Mahalanobis laten","0,81633","−0,00659","1,000","lebih buruk","w"],
  ["allnum_anomaly","Anomali semua fitur numerik","0,81429","−0,00862","1,000","lebih buruk","w"],
  ["blockwise_ae","AE per-blok korelasi V","0,81314","−0,00978","1,000","lebih buruk","w"],
  ["missingness_ae","Embedding pola missing","0,81131","−0,01161","1,000","lebih buruk","w"],
  ["concat_latent","V asli + 32 laten AE","0,79851","−0,02440","1,000","lebih buruk","w"],
  ["sae_latent","Supervised AE laten (V)","0,76880","−0,05412","1,000","lebih buruk","w"],
  ["perfeat_anomaly","Error per-fitur (339 dim)","0,73818","−0,08474","1,000","lebih buruk","w"],
  ["sae_allnum","Supervised AE semua numerik","0,73355","−0,08937","1,000","lebih buruk","w"],
];
const colW = [1700, 2900, 1300, 1500, 900, 1060];
function cell(text, w, opts={}) {
  return new TableCell({ borders, width:{size:w,type:WidthType.DXA},
    shading: opts.fill ? {fill:opts.fill,type:ShadingType.CLEAR} : undefined,
    margins:{top:60,bottom:60,left:90,right:90},
    children:[new Paragraph({children:[new TextRun({text,bold:opts.bold||false,size:opts.size||18})]})]});
}
function headerRow(){
  const labels=["Varian","Deskripsi","PR-AUC","Δ vs base","p","Verdict"];
  return new TableRow({tableHeader:true,children:labels.map((l,i)=>cell(l,colW[i],{bold:true,fill:"1F3B57",size:18}))});
}
function colored(t){return t==="tie"?"FDF3D0":"F7E3E0";}
const table = new Table({ width:{size:CW,type:WidthType.DXA}, columnWidths:colW,
  rows:[ headerRow(),
    new TableRow({children:[
      cell("baseline",colW[0],{bold:true,fill:"DCEAF5"}),cell("LightGBM fitur asli (acuan)",colW[1],{fill:"DCEAF5"}),
      cell("0,82292",colW[2],{bold:true,fill:"DCEAF5"}),cell("—",colW[3],{fill:"DCEAF5"}),
      cell("—",colW[4],{fill:"DCEAF5"}),cell("acuan",colW[5],{fill:"DCEAF5"})]}),
    ...masterRows.map(r=>new TableRow({children:[
      cell(r[0],colW[0],{fill:colored(r[6]),bold:r[6]==="tie"}),cell(r[1],colW[1],{fill:colored(r[6])}),
      cell(r[2],colW[2],{fill:colored(r[6])}),cell(r[3],colW[3],{fill:colored(r[6])}),
      cell(r[4],colW[4],{fill:colored(r[6])}),cell(r[5],colW[5],{fill:colored(r[6])})]}))
  ]});

const doc = new Document({
  styles:{ default:{document:{run:{font:"Arial",size:22}}},
    paragraphStyles:[
      {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,
       run:{size:30,bold:true,font:"Arial",color:"1F3B57"},paragraph:{spacing:{before:280,after:160},outlineLevel:0}},
      {id:"Heading2",name:"Heading 2",basedOn:"Normal",next:"Normal",quickFormat:true,
       run:{size:25,bold:true,font:"Arial",color:"2E5070"},paragraph:{spacing:{before:200,after:120},outlineLevel:1}},
    ]},
  numbering:{config:[{reference:"b",levels:[{level:0,format:LevelFormat.BULLET,text:"•",alignment:AlignmentType.LEFT,
    style:{paragraph:{indent:{left:560,hanging:280}}}}]}]},
  sections:[{
    properties:{page:{size:{width:12240,height:15840},margin:{top:1440,right:1440,bottom:1440,left:1440}}},
    children:[
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:60},
        children:[new TextRun({text:"Kajian Penyebab dan Eksplorasi Perbaikan",bold:true,size:34,color:"1F3B57"})]}),
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:60},
        children:[new TextRun({text:"Pendekatan Autoencoder sebagai Penyedia Fitur untuk LightGBM",bold:true,size:26,color:"2E5070"})]}),
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:240},
        children:[new TextRun({text:"Deteksi Fraud pada Dataset IEEE-CIS  •  20 Juni 2026",size:20,color:"555555"})]}),

      h1("1. Latar Belakang dan Tujuan"),
      p("Usulan tugas akhir ini menempatkan autoencoder (AE) sebagai komponen yang memperbaiki/menyediakan fitur bagi LightGBM untuk mendeteksi transaksi fraud pada dataset IEEE-CIS. Pada implementasi awal, desain tersebut (AE merekonstruksi dan menggantikan blok fitur V) ternyata tidak meningkatkan performa, bahkan menurunkannya."),
      p("Mengikuti arahan pembimbing, kajian ini tidak berpindah ke metode lain, melainkan: (1) tetap pada tujuan usulan, (2) mendiagnosis penyebab kekurangan secara empiris, dan (3) mengeksplorasi perbaikan yang tetap berada dalam koridor tujuan tersebut."),
      p("Protokol evaluasi konsisten di seluruh kajian: pembagian data berstrata (train/validasi/uji 60/20/20) dengan random_state 42, metrik utama PR-AUC (Average Precision), pemilihan ambang batas pada data validasi (berdasarkan MCC), dan uji signifikansi paired bootstrap 2000 resampling.",{italics:true,size:18}),

      h1("2. Diagnosis Penyebab"),
      p("Desain awal (mengganti blok V dengan rekonstruksi AE) menurunkan PR-AUC dari 0,82184 (baseline) menjadi 0,76896 (Δ −0,0529; p = 1,000). Untuk menguji penyebabnya, dilakukan sweep ukuran dimensi laten AE sambil mengukur kualitas rekonstruksi (R²) dan PR-AUC."),
      img("diagnosis_replace_v.png", 520, 300),
      cap("Gambar 1. Kualitas rekonstruksi (R²) naik 0,90→0,95 seiring membesarnya dimensi laten, tetapi Δ PR-AUC tetap ~−0,035 (tidak pulih)."),
      h2("Temuan diagnosis"),
      bullet("Rekonstruksi yang makin akurat TIDAK memulihkan PR-AUC — jadi penyebabnya bukan sekadar kompresi terlalu kecil."),
      bullet("Autoencoder meminimalkan rata-rata error rekonstruksi, sehingga mempertahankan pola umum namun menghaluskan deviasi-deviasi halus pada fitur V yang justru menjadi penanda fraud (kelas langka). Rekonstruksi AE berperilaku seperti low-pass filter."),
      bullet("Fitur V hanya menyumbang ~22,4% importance pada baseline, namun sudah dimanfaatkan optimal oleh LightGBM. Analisis missingness menunjukkan pola missing yang sedikit (≈6 pola dominan) dan sudah ditangani LightGBM secara native — sehingga tidak menyisakan struktur baru yang berarti bagi AE."),

      h1("3. Eksplorasi Perbaikan (14 Pendekatan)"),
      p("Sebanyak 14 pendekatan autoencoder yang berbeda diuji pada FULL DATA, seluruhnya tetap dalam koridor “AE menyediakan fitur untuk LightGBM”. Cakupan meliputi: fitur laten, error rekonstruksi (agregat, per-fitur, one-class normal, fraud, kontras), supervised autoencoder, jarak laten, variational AE, embedding pola missing, hybrid IsolationForest, AE per-blok, dan stacking neural."),
      img("tabel_master_14varian.png", 660, 410),
      cap("Tabel 1. Perbandingan 14 pendekatan AE-fitur terhadap baseline (seluruh data, pembagian berstrata)."),
      table,
      new Paragraph({spacing:{before:120}}),

      p("Catatan: dari 14 pendekatan per-baris ini, tidak ada yang mengalahkan baseline; terbaik (ae_mlp_stack) hanya seri (Δ −0,00118; p = 0,807). Kesamaan mereka: seluruhnya bekerja per-baris pada fitur yang sudah dimiliki LightGBM, sehingga bersifat redundan. Hal ini mengarahkan pada hipotesis bahwa autoencoder hanya berpeluang membantu bila diberi informasi yang tidak dapat diturunkan LightGBM dari satu baris — yakni informasi relasional/antar-transaksi.",{italics:true,size:18}),

      h1("4. Temuan Relasional dan Kontrol Atribusi"),
      p("Berdasarkan hipotesis di atas, diuji dua pendekatan yang memberi LightGBM informasi relasional/entitas: (a) entity_ae — autoencoder memampatkan profil agregat per entitas (count dan statistik nominal transaksi per card1/addr1/email/perangkat, tanpa label); dan (b) cat_embedding — embedding terpelajar untuk kategori berkardinalitas tinggi. Keduanya untuk pertama kalinya MENGALAHKAN baseline secara signifikan (entity_ae +0,0171; cat_embedding +0,0188; keduanya p < 0,001)."),
      p("Namun, untuk menentukan apakah kenaikan berasal dari autoencoder atau dari fitur relasional yang mendasarinya, dilakukan kontrol adil tanpa AE: (a) agregat entitas mentah langsung ke LightGBM, dan (b) target encoding OOF untuk kategori yang sama."),
      img("atribusi_relasional.png", 540, 300),
      cap("Gambar 2. Perbandingan usulan (AE/embedding) vs kontrol (tanpa AE). Semua p < 0,001."),
      (function(){
        const cw=[2600,2400,1400,1480,1480];
        const cell=(t,w,o={})=>new TableCell({borders,width:{size:w,type:WidthType.DXA},
          shading:o.fill?{fill:o.fill,type:ShadingType.CLEAR}:undefined,
          margins:{top:60,bottom:60,left:90,right:90},
          children:[new Paragraph({children:[new TextRun({text:t,bold:o.bold||false,size:o.size||18})]})]});
        const hr=new TableRow({tableHeader:true,children:["Aspek","Usulan (AE/embedding)","Kontrol (tanpa AE)","Selisih","Pemenang"]
          .map((l,i)=>cell(l,cw[i],{bold:true,fill:"1F3B57",size:18}))});
        const r1=new TableRow({children:[cell("Relasional (profil entitas)",cw[0]),cell("entity_ae  +0,0171",cw[1]),
          cell("entity_raw  +0,0207",cw[2],{fill:"F7E3E0"}),cell("kontrol +0,0036",cw[3]),cell("Kontrol (AE tak berkontribusi)",cw[4],{fill:"F7E3E0",bold:true})]});
        const r2=new TableRow({children:[cell("Kategorikal (kardinalitas tinggi)",cw[0]),cell("cat_embedding  +0,0188",cw[1],{fill:"DDEFDD"}),
          cell("target_encode  +0,0090",cw[2]),cell("usulan +0,0098",cw[3]),cell("Usulan (embedding supervised)",cw[4],{fill:"DDEFDD",bold:true})]});
        return new Table({width:{size:CW,type:WidthType.DXA},columnWidths:cw,rows:[hr,r1,r2]});
      })(),
      new Paragraph({spacing:{before:120}}),
      p("Hasil kontrol bersifat menentukan: pada fitur relasional, agregasi entitas MENTAH (+0,0207) justru lebih unggul daripada versi yang dimampatkan autoencoder (+0,0171) — artinya kenaikan berasal dari fitur relasionalnya, bukan dari autoencoder, yang bahkan sedikit merugikan karena bersifat lossy. Pada sisi kategorikal, embedding (+0,0188) mengungguli target encoding (+0,0090), namun embedding tersebut dilatih secara supervised dan tidak melakukan rekonstruksi sehingga secara teknis bukan autoencoder."),

      h1("5. Kesimpulan"),
      p("Pada dataset IEEE-CIS, autoencoder tidak meningkatkan performa LightGBM sebagai penyedia fitur. Kesimpulan ini dibuktikan secara menyeluruh: 14 pendekatan AE per-baris gagal (termasuk diagnosis sebab), dan pada pendekatan relasional yang sempat unggul, kontrol menunjukkan bahwa agregasi entitas sederhana mengungguli autoencoder.",{bold:true}),
      p("Pengungkit performa yang sebenarnya adalah informasi relasional/entitas — titik buta LightGBM yang memproses tiap transaksi secara terpisah — dan paling efektif ditangkap melalui agregasi fitur entitas, bukan melalui autoencoder. Representasi terpelajar yang menambah nilai pada sisi kategorikal pun berupa embedding supervised, bukan autoencoder."),
      p("Dengan demikian, keterbatasan autoencoder bersifat fundamental terhadap karakter data IEEE-CIS (fitur telah direkayasa Vesta; nilai hilang dan pola missing telah ditangani LightGBM secara native), dan analisis ini telah mengatribusikan sumber kenaikan performa secara adil melalui kontrol terkendali."),

      h1("6. Rekomendasi"),
      bullet("Melaporkan hasil sebagai temuan yang sah, menyeluruh, dan terkontrol: pertanyaan penelitian terjawab dengan atribusi yang jelas (AE tidak berkontribusi; fitur relasional yang berperan)."),
      bullet("Mendiskusikan arah lanjutan dengan pembimbing: (a) menjadikan studi keterbatasan AE sebagai inti, dengan temuan relasional sebagai analisis pendukung; atau (b) bila berkenan, mengembangkan fitur relasional/entitas sebagai kontribusi utama."),
      bullet("Sebagai penyempurnaan metodologis, menyelaraskan protokol embedding kategorikal ke skema out-of-fold agar perbandingan dengan target encoding sepenuhnya setara."),
      bullet("Menjadikan evaluasi pada protokol temporal/realistis dan isu concept drift sebagai future work."),
    ],
  }],
});

Packer.toBuffer(doc).then(b=>{fs.writeFileSync(OUT,b);console.log("Saved:",OUT);});
