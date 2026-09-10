import unittest

from itn_v2.normalizers.identifier import (AddressParser, CallsignParser,
                                           DocumentIDParser, IMOParser,
                                           LegalDocumentParser, MMSIParser,
                                           PortCodeParser, TelephoneParser,
                                           VehiclePlateParser)
from itn_v2.validators import validate


class TestIdentifiers(unittest.TestCase):
    def test_mmsi_dung_chin_chu_so(self):
        r = MMSIParser()("hai ba bốn năm sáu bảy tám chín không")
        self.assertEqual(r.normalized, "234567890")
        self.assertTrue(validate("MMSI_ID", r.normalized)[0])

    def test_mmsi_sai_do_dai_bi_validator_chan(self):
        r = MMSIParser()("hai ba bốn năm sáu bảy tám chín")
        self.assertFalse(validate("MMSI_ID", r.normalized)[0])

    def test_telephone(self):
        r = TelephoneParser()("không chín không tám một hai ba bốn năm sáu")
        self.assertEqual(r.normalized, "0908123456")
        self.assertTrue(validate("TELEPHONE", r.normalized)[0])

    def test_telephone_doc_theo_nhom(self):
        roi = TelephoneParser()("không chín không tám một hai ba bốn năm sáu").normalized
        nhom = TelephoneParser()("không chín không tám mười hai ba bốn năm sáu").normalized
        self.assertEqual(len(roi), len(nhom))

    def test_imo(self):
        r = IMOParser()("i em ô chín không bảy bốn bảy hai chín")
        self.assertEqual(r.normalized, "IMO 9074729")
        self.assertTrue(validate("IMO_ID", r.normalized)[0])

    def test_port_code_tu_danh_muc(self):
        self.assertEqual(PortCodeParser()("vê en hát pê hát").normalized, "VNHPH")

    def test_port_code_ngoai_danh_muc_van_ghep_duoc(self):
        r = PortCodeParser()("ét giê ét i en")
        self.assertEqual(r.normalized, "SGSIN")

    def test_document_id(self):
        self.assertEqual(DocumentIDParser()("số mười hai xẹt hai không hai tư").normalized,
                         "Số 12/2024")

    def test_legal_doc(self):
        self.assertEqual(LegalDocumentParser()("nờ đê cê pê").normalized, "NĐ-CP")
        self.assertEqual(LegalDocumentParser()("tê tê bê tê xê").normalized, "TT-BTC")

    def test_legal_doc_kem_so_hieu(self):
        self.assertEqual(
            LegalDocumentParser()("số mười hai xẹt hai không hai tư nờ đê cê pê").normalized,
            "Số 12/2024/NĐ-CP")

    def test_legal_doc_khong_tra_duoc_thi_truot(self):
        self.assertFalse(LegalDocumentParser()("ích ích ích").valid)

    def test_callsign(self):
        self.assertEqual(CallsignParser()("ba vê hát a").normalized, "3VHA")

    def test_vehicle_plate(self):
        r = VehiclePlateParser()("ba mươi a một hai ba bốn năm")
        self.assertEqual(r.normalized, "30A-12345")
        self.assertTrue(validate("VEHICLE_PLATE", r.normalized)[0])

    def test_vehicle_plate_sai_khuon(self):
        self.assertFalse(VehiclePlateParser()("a bê xê").valid)

    def test_address(self):
        self.assertEqual(AddressParser()("số mười hai đường Lê Lợi").normalized,
                         "Số 12 đường Lê Lợi")


if __name__ == "__main__":
    unittest.main()
